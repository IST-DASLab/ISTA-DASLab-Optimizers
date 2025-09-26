class GlobalCache:
    _instance = None

    @staticmethod
    def init():
        if GlobalCache._instance is None:
            GlobalCache._instance = GlobalCache()

    @staticmethod
    def contains(category, key):
        return GlobalCache.get_instance()._contains(category, key)

    @staticmethod
    def get_instance():
        if GlobalCache._instance is None:
            GlobalCache.init()
        return GlobalCache._instance

    @staticmethod
    def get(category, key):
        return GlobalCache.get_instance()._get(category, key)

    @staticmethod
    def add(category, key, item):
        return GlobalCache.get_instance()._add(category, key, item)

    def __init__(self):
        self.storage = dict()

    def _contains(self, category, key):
        if category not in self.storage:
            return False
        return key in self.storage[category]

    def _add(self, category, key, item):
        if category not in self.storage:
            self.storage[category] = { key: item }
        elif key not in self.storage[category]:
            self.storage[category][key] = item

    def _get(self, category, key):
        # print(self.storage)
        if self._contains(category, key):
            return self.storage[category][key]
        raise ValueError(f'GlobalCache does not contain category {category} and/or key {key}')