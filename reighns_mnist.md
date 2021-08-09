

## config

### mkdir
1. Create directories for us if it does not exists, so whenever you import config file, this will execute once.

### mlflow set_tracking_uri

1. dd


### logging_config

1. handler: piece of code that does the logging.
2. Prints out error message in console.
   ```
           "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "minimal",
            "level": logging.DEBUG,
        },
    ```
3. Writes messahes to `LOGS_DIR\info.log`. Here the level of severity of handler is at INFO level, anything above this level will be logged. As an example, WARNING, ERROR, CRITICAL are levels above INFO, and hence will be logged. Same logic applies to the keys "error".
    ```
            "info": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "info.log"),
            "maxBytes": 10485760,  # 1 MB
            "backupCount": 10,
            "formatter": "detailed",
            "level": logging.INFO,
        },
    ```
4. Combining a set of handlers and log. You can refer to explanation in https://madewithml.com/courses/mlops/logging/.

