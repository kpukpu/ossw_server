# Generated by Django 4.2.11 on 2024-05-30 07:21

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0002_alter_user_model_firstname_alter_user_model_username"),
    ]

    operations = [
        migrations.DeleteModel(name="user_model",),
    ]
