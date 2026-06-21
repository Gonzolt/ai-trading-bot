FROM quantumtrader-base:latest
CMD ["celery", "-A", "quantumtrader.scheduler:celery_app", "beat", "--loglevel=INFO"]
