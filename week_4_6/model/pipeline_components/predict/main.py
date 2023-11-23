#### PUT YOUR CODE HERE ####
# This module is intended to be used in the serving container
# import libraries

logger = logging.getLogger("App")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def load_model(path: str):
    """Loads a model artifact"""
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


def model_predict(model: Dict, data: List[Dict[str, Dict]]) -> List:
    #### PUT YOUR CODE HERE ####
    return  #### PUT YOUR CODE HERE ####
            #### EXAMPLE: return list(pipeline.predict(pd.DataFrame([list(x.values())[0] for x in data]))