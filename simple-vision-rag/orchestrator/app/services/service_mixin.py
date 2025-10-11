from fastapi import HTTPException


class ServiceMixin:
    def _handle_error(self, e: Exception):
        raise HTTPException(
            status_code=500, detail=f"An error occurred during request: {e}"
        )
