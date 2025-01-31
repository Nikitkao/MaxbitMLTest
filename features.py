class Features:
    def __init__(self):
        self.num_features = ['tree_dbh']  # Только tree_dbh из числовых
        self.cat_features = [
            'spc_latin', 'spc_common', 'postcode', 'borough', 'zip_city', 'steward', 'guards', 'sidewalk', 'user_type',
            'root_stone', 'root_grate', 'root_other', 'trunk_wire', 'trnk_light', 'trnk_other', 'brch_light', 'brch_shoe', 'brch_other', 'curb_loc'
        ]  # Категориальные признаки

    def get_num_features(self):
        return self.num_features

    def get_cat_features(self):
        return self.cat_features