from pydantic_settings import BaseSettings, SettingsConfigDict


language = "ru"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="dev.env", env_file_encoding="utf-8", extra="ignore")
    LOG_LEVEL: str = "DEBUG"
    APP_DB_USER: str
    APP_DB_PASS: str
    APP_DB_HOST: str
    APP_DB_NAME: str
    APP_DB_PORT: str
    APIKEY: str
    SECRETKEY: str

    @property
    def APP_DB_URL(self):
        db_url = "mssql+pymssql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        return db_url.format(
            db_user=self.APP_DB_USER,
            db_pass=self.APP_DB_PASS,
            db_host=self.APP_DB_HOST,
            db_instance=self.APP_DB_PORT,
            db_name=self.APP_DB_NAME,
        )
