FROM quantumtrader-base:latest
EXPOSE 8000
CMD ["uvicorn", "quantumtrader.dashboard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
