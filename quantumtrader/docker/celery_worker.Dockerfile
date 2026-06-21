FROM quantumtrader-base:latest
CMD ["celery", "-A", "quantumtrader.scheduler:celery_app", "worker", "--loglevel=INFO"]
