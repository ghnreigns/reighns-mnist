1. config.py: This will create all the folders needed. Which begs the question on when should I execute this script?
2. config.py: mlflow.set_tracking_uri(uri="file://" + str(MODEL_REGISTRY.absolute())) I temporary commented this line because it affects my call in the main file.
3. Vs code is complaining imports because in the original environment, there wasnt this module.
4. 