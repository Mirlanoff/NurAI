class NurAIError(Exception):
    pass


class EmptyDocumentError(NurAIError):
    pass


class UploadTooLargeError(NurAIError):
    pass


class VectorStoreUnavailableError(NurAIError):
    pass
