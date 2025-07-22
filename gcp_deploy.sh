#!/bin/bash
set -euo pipefail

source ./.env.gcp

# définition des variables

REGION="europe-west1"
ZONE="europe-west1-b"
PROJECT_ID="chainlit-rag-juridique"
KEY_FILE="gcp_key.json"
CHAINLIT_APP_NAME="chainlit-app"
ARTIFACTS_REPO_NAME="chainlit-repo"
COMPUTE_INSTANCE="qdrant-vm"
FIREWALL_RULME_NAME="allow-qdrant"

POSTGRES_SERVER_NAME=chainlit-postgres
CHAINLIT_DATABASE_PORT=$CHAINLIT_DATABASE_PORT
CHAINLIT_DATABASE_NAME=$CHAINLIT_DATABASE_NAME
CHAINLIT_DATABASE_USERNAME=$CHAINLIT_DATABASE_USERNAME
CHAINLIT_DATABASE_ROOT_PASSWORD=$CHAINLIT_DATABASE_ROOT_PASSWORD
CHAINLIT_AUTH_SECRET=$CHAINLIT_AUTH_SECRET

QDRANT_API_KEY=$QDRANT_API_KEY

MODEL_TYPE=$MODEL_TYPE
GROQ_API_KEY=$GROQ_API_KEY
GROQ_MODEL=$GROQ_MODEL
CLOUD_OLLAMA_EMBEDDING_MODEL=$CLOUD_OLLAMA_EMBEDDING_MODEL


######################################################################

# suppression des instances deja créées
echo "=================================================================================="
echo "suppression des instances si elles existent"
echo "=================================================================================="

gcloud artifacts repositories delete $ARTIFACTS_REPO_NAME --location=$REGION --quiet || true
gcloud sql databases delete $CHAINLIT_DATABASE_NAME --instance=$POSTGRES_SERVER_NAME --quiet || true
gcloud sql instances delete $POSTGRES_SERVER_NAME --quiet || true
gcloud compute instances delete $COMPUTE_INSTANCE --zone=$ZONE --quiet || true
gcloud compute firewall-rules delete $FIREWALL_RULME_NAME --quiet || true
gcloud run jobs delete chainlit-migrate --region=$REGION --quiet || true
gcloud run services delete $CHAINLIT_APP_NAME --region=$REGION --quiet || true

######################################################################

# connexion au projet GCP
echo "=================================================================================="
echo "connexion au projet GCP et configuration"
echo "=================================================================================="

gcloud auth activate-service-account --key-file=$KEY_FILE

# configuration du projet et de la région
gcloud config set project $PROJECT_ID --quiet
gcloud config set compute/region $REGION --quiet
gcloud config set compute/zone $ZONE --quiet

# activation des services : 
gcloud services enable compute.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable sql-component.googleapis.com
gcloud services enable sqladmin.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable secretmanager.googleapis.com

# créer un dépot pour stocker les images docker
echo "=================================================================================="
echo "création dun dépot pour stocker les images docker"
echo "=================================================================================="

gcloud artifacts repositories create $ARTIFACTS_REPO_NAME \
  --repository-format=docker \
  --location=$REGION \
  --description="Images Docker pour Chainlit RAG"

# créer la base de données postgres avec Cloud SQL
echo "=================================================================================="
echo "création de l'instance SQLpostgres avec Cloud SQL"
echo "=================================================================================="

gcloud sql instances create $POSTGRES_SERVER_NAME \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --region=$REGION \
  --storage-type=SSD \
  --storage-size=10GB \
  --authorized-networks=0.0.0.0/0

# créer la base de données
echo "=================================================================================="
echo "création de la base de données"
echo "=================================================================================="

gcloud sql databases create $CHAINLIT_DATABASE_NAME --instance=$POSTGRES_SERVER_NAME

# créer l'utilisateur de la base de données
echo "=================================================================================="
echo "création l'utilisateur de la base de données"
echo "=================================================================================="

gcloud sql users create $CHAINLIT_DATABASE_USERNAME --instance=$POSTGRES_SERVER_NAME --password=$CHAINLIT_DATABASE_ROOT_PASSWORD

# récupérer l'adresse IP de la base de données
DB_IP=$(gcloud sql instances describe $POSTGRES_SERVER_NAME --format='value(ipAddresses[0].ipAddress)')
CHAINLIT_DATABASE_HOST=$DB_IP

# créer une VM pour Qdrant
echo "=================================================================================="
echo "création  d'une VM pour Qdrant avec compute"
echo "=================================================================================="

gcloud compute instances create $COMPUTE_INSTANCE \
  --zone=$ZONE \
  --machine-type=e2-standard-2 \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=100GB \
  --tags=qdrant-server

# récupérer l'adresse IP de la VM
VM_IP=$(gcloud compute instances describe $COMPUTE_INSTANCE --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
QDRANT_HOST=http://$VM_IP:6333

sleep 60

# ouverture des ports nécessaires
echo "=================================================================================="
echo "ouverture des ports nécessaires"
echo "=================================================================================="

gcloud compute firewall-rules create $FIREWALL_RULME_NAME \
  --allow tcp:6333,tcp:6334 \
  --source-ranges=0.0.0.0/0 \
  --target-tags=qdrant-server

# installation de docker sur la VM et deploiement de Qdrant
echo "=================================================================================="
echo "installation de docker sur la VM et deploiement de Qdrant"
echo "=================================================================================="

gcloud compute ssh $COMPUTE_INSTANCE --zone=$ZONE --command="
    curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh && sudo usermod -aG docker $USER"
gcloud compute instances reset $COMPUTE_INSTANCE --zone=$ZONE
sleep 60

# transfert des données Qdrant locales vers la VM
echo "=================================================================================="
echo "transfert des données Qdrant locales vers la VM"
echo "=================================================================================="

gcloud compute scp --recurse ./qdrant_storage $COMPUTE_INSTANCE:/home/$(whoami)/ --zone=$ZONE

# lancement du container qdrant sur la VM
echo "=================================================================================="
echo "lancement du container qdrant sur la VM"
echo "=================================================================================="

gcloud compute ssh $COMPUTE_INSTANCE --zone=$ZONE --command="
sudo docker run -d \
    --name qdrant \
    -p 6333:6333 -p 6334:6334 \
    -v ./qdrant_storage:/qdrant/storage \
    -e QDRANT__SERVICE__API_KEY=$QDRANT_API_KEY \
    qdrant/qdrant:latest"

echo "=================================================================================="
echo "construire et pousser l'appli sur Artifact Registry"
echo "=================================================================================="

# configurer l'authentification docker pour utiliser Artifact Registry
gcloud auth configure-docker $REGION-docker.pkg.dev    

# construire et pousser l'appli sur Artifact Registry
if ! docker build -f Dockerfile.chainlit -t chainlit-app .; then
    echo "Échec du build Docker ! Script arrêté."
    exit 1
fi
docker tag chainlit-app:latest $REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACTS_REPO_NAME/chainlit-app:latest
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACTS_REPO_NAME/chainlit-app:latest

# deploiement de l'app sur Cloud Run
echo "=================================================================================="
echo "migrations avant deploiement"
echo "=================================================================================="

# migrations avant deploiement
gcloud run jobs create chainlit-migrate --region=$REGION \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACTS_REPO_NAME/chainlit-app:latest \
  --command "bash" \
  --args "-c,cd src/chainlit_app && uv run alembic upgrade head && cd /app && uv run python create_admin.py" \
  --set-env-vars "CHAINLIT_DATABASE_HOST=$DB_IP,CHAINLIT_DATABASE_PORT=$CHAINLIT_DATABASE_PORT,CHAINLIT_DATABASE_NAME=$CHAINLIT_DATABASE_NAME,CHAINLIT_DATABASE_USERNAME=$CHAINLIT_DATABASE_USERNAME,CHAINLIT_DATABASE_ROOT_PASSWORD=$CHAINLIT_DATABASE_ROOT_PASSWORD"

# lancement du job de migration
gcloud run jobs execute chainlit-migrate --region=$REGION

# déploiement chainlit
echo "=================================================================================="
echo "déploiement chainlit"
echo "=================================================================================="

gcloud run deploy $CHAINLIT_APP_NAME \
  --image $REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACTS_REPO_NAME/chainlit-app:latest \
  --platform managed \
  --region $REGION \
  --port 8000 \
  --memory 6Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10 \
  --allow-unauthenticated \
  --set-env-vars "MODEL_TYPE=$MODEL_TYPE,CHAINLIT_DATABASE_HOST=$CHAINLIT_DATABASE_HOST,CHAINLIT_DATABASE_PORT=$CHAINLIT_DATABASE_PORT,CHAINLIT_DATABASE_NAME=$CHAINLIT_DATABASE_NAME,CHAINLIT_DATABASE_USERNAME=$CHAINLIT_DATABASE_USERNAME,CHAINLIT_DATABASE_ROOT_PASSWORD=$CHAINLIT_DATABASE_ROOT_PASSWORD,QDRANT_HOST=$QDRANT_HOST,GROQ_API_KEY=$GROQ_API_KEY,GROQ_MODEL=$GROQ_MODEL,CLOUD_OLLAMA_EMBEDDING_MODEL=$CLOUD_OLLAMA_EMBEDDING_MODEL,QDRANT_API_KEY=$QDRANT_API_KEY,CHAINLIT_AUTH_SECRET=$CHAINLIT_AUTH_SECRET"

# obtenir l'url publique de l'appli chainlit
echo "=================================================================================="
echo "url publique de l'appli chainlit"
echo "=================================================================================="

echo "L'application Chainlit est déployée et accessible à l'adresse :"
echo
gcloud run services list --region=europe-west1 --filter="SERVICE:$CHAINLIT_APP_NAME" --format="value(URL)"
