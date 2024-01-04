from django.apps import AppConfig
from main.cd_model.inference import DHJModel


class MainConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'main'

    model = DHJModel("inference")