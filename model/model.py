class RecSysMF():
    def __init__(self):
        self.model = None
        self.users_matrix = None
        self.items_matrix = None
        self.ratings_train = None
        self.ratings_test = None
        self.user_item_matrix = None
        self.mean_user_rating = None
        # self.std_user_rating = None
        self.n_users = None
        self.n_items = None
        self.trained = False
        self.surprise_matrix = None
        self.items_similarity_matrix = None

    def train(self):
        return 'trian'

    def evaluate(self):
        return 'evaluate'

    def predict(self):
        return 'predict'
