from abc import ABC, abstractmethod


settings = {
    'classification': 'models/classification_model.pkl',
    'segmentation': {'all': ['models/all/cascade1.pkl', 'models/all/cascade2.pkl'], 'cross': ['models/cross/cascade1.pkl', 'models/cross/cascade2.pkl'], 'long': ['models/long/cascade1.pkl', 'models/long/cascade2.pkl']},
    'tracking': 'models/tracking_model.pkl',
}


class ModelABC(ABC):

    def __init__(self):
        self._model = None
        # self.load(settings['BASE_PATH'])

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Функция, в которой обределяется структура NN и
        происходит загрузка весов модели в self._model

        params:
          path - путь к файлу, в котором содержатся веса модели
        """
        ...

    @abstractmethod
    def preprocessing(self, path: str) -> object:
        """
        Функция, котороя предобрабатывает изображение к виду, 
        с которым можеn взаимодействовать модель из self._model

        params:
          path - путь к файлу (изображению .tiff/.png), который будет
                использоваться для предсказания

        return - возвращает предобработанное изображение 
        """
        ...

    @abstractmethod
    def predict(self, path: str) -> object:
        """
        Функция, в которой предобработанное изображение подается
        на входы NN (self._model) и возвращается результат работы NN 

        params:
          path - путь к файлу (изображению .tiff/.png), который будет
                использоваться для предсказания

        return - результаты предсказания
        """
        ...


class YourModel(ModelABC):

    def load(self, path: str) -> None:
        # Пример на псевдо-питоновском
        # with open(path, 'rb') as inp:
        #   weights = inp.read()
        #   self._model = nn.Pypeline(
        #     nn.Layer1(weights=weights[0]),
        #     nn.Layer2(weights=weights[1]),
        #   )
        pass

    def preprocessing(self, path: str) -> object:
        # Пример на псевдо-питоновском
        # with open(path, 'rb') as inp:
        #   x = nn.Tiff2Image(inp)
        #   preproced_x = nn.normalize(x)
        # return preproced_x
        pass

    def predict(self, path: str) -> object:
        # Пример на псевдо-питоновском
        # x = self.preprocessing(path)
        # return self._model.predict(x)
        pass
