"""
自定义异常类
用于服务层统一抛出业务异常
"""

from fastapi import HTTPException, status


class APIException(HTTPException):
    """API 基础异常类"""

    def __init__(
        self,
        status_code: int,
        detail: str,
        code: str = "error",
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.code = code


class NotFoundException(APIException):
    def __init__(self, detail: str = "请求的资源不存在"):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            code="not_found",
        )


class ValidationException(APIException):
    def __init__(self, detail: str = "数据验证失败"):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=detail,
            code="validation_error",
        )


class ServiceException(APIException):
    def __init__(self, detail: str = "服务暂时不可用"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            code="service_error",
        )


class ServiceUnavailableException(APIException):
    def __init__(self, detail: str = "服务暂时不可用，请稍后重试"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            code="service_unavailable",
        )
