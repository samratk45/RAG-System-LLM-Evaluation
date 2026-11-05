import sys
import traceback
from typing import Optional


class DocumentPortalException(Exception):
    def __init__(self, error_message, error_details: Optional[object] = None):
        self.error_message = str(error_message)

        if isinstance(error_details, BaseException):
            exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__
        elif error_details is not None and hasattr(error_details, "exc_info"):  # e.g. the sys module
            exc_type, exc_value, exc_tb = error_details.exc_info()
        else:
            exc_type, exc_value, exc_tb = sys.exc_info()

        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.traceback_str = (
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            if exc_type and exc_tb else ""
        )
        super().__init__(str(self))

    def __str__(self):
        return f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"

    def __repr__(self):
        return (f"DocumentPortalException(file={self.file_name!r}, "
                f"line={self.lineno}, message={self.error_message!r})")