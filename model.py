from abc import abstractmethod


class BaseModel:
    @abstractmethod
    def get_model(self):
        pass
