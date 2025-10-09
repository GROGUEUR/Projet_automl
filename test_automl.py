from automl import AutoML

# Chemin vers les données du projet
data_path = "/info/corpus/ChallengeMachineLearning"

# Utiliser votre AutoML
model = AutoML()
model.fit(data_path)
model.eval()
