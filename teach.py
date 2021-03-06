import asyncio

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from db import setup_mongo
from settings import config
from utils import save_model, load_model as sync_load_model

MODEL_FILE_NAME = 'filename.pickle'


def train(list_x, list_y):
    NUMBER_OF_LAYERS = 3
    NEURONS_PER_LAYER = 80

    classifier = MLPClassifier(
        hidden_layer_sizes=((1, 256), (2, 64), (3, 16)),
        alpha=0.001,
        random_state=1
    )

    pipeline = Pipeline([
        ('classifier', classifier)
    ])

    model = pipeline.fit(X=list_x, y=list_y)
    return model


async def train_and_save(collection):
    """
    Train your model and save it for later.

    The template of the function used for training the model.

    You need to write the code reponsible for data
    preprocessing and training the model of ML algorithm you chose.
    """

    # Create you model - choose any classification algorithm you wish.

    # Extract features and labels from tha data in the database.
    cursor = collection.find({})
    last_distances = [0]
    input_series = []
    output_series = []
    async for document in cursor:
        data = document['data']
        input_series.append([
            data['yV'],
            data['hV'],
            *[
                d if d is not None else 30
                for d in data['s']
            ],
            data['x'],
            last_distances[0]
        ])
        output_series.append([
            data['u'],
            data['l'],
            data['r'],
        ])
        last_distances.append(data['x'])
        last_distances = last_distances[:60 * 5]

    NUMBER_OF_LAYERS = 1
    NEURONS_PER_LAYER = 20

    classifier = MLPClassifier(
        hidden_layer_sizes=(NEURONS_PER_LAYER,) * NUMBER_OF_LAYERS,
        alpha=0.01,
        random_state=1
    )

    pipeline = Pipeline([
        ('classifier', classifier)
    ])

    model = pipeline.fit(X=input_series, y=output_series)

    save_model(model, MODEL_FILE_NAME)


last_distances = [0]
max_distance = 0
turns_since_max = 0
revert_turns = 0
prev_datas = []


async def load_model():
    """
    Load a trained model that you prepared in `train_and_save` earlier.
    """

    # The game will not start unless this callback finishes, take your
    # time to load/compile the model now.
    model = sync_load_model(MODEL_FILE_NAME)

    global last_distances, max_distance, turns_since_max, revert_turns, prev_datas

    last_distances = [0]
    max_distance = 0
    turns_since_max = 0
    prev_datas = []
    revert_turns = 0

    return model


async def predict(model, yV, hV, s, x, ts):
    """
    Make predictions during a game.

    Given incoming data return an iterable of boolean values indicating
    which controls should be active.

    Should return a tuple of booleans (PRESS_UP, PRESS_LEFT, PRESS_RIGHT)
    """

    global last_distances, max_distance, turns_since_max, revert_turns, prev_datas

    data = [
        yV,
        hV,
        *[
            d if d is not None else 30
            for d in s
        ],
        x,
        last_distances[0],
    ]
    prev_datas.append(data)
    prev_datas = prev_datas[-10:]
    result = model.predict(X=prev_datas)

    last_distances.append(x)
    last_distances = last_distances[:60 * 5]

    while revert_turns > 0:
        revert_turns -= 1
        return 1 - result[0][0], 1 - result[0][1], 1 - result[0][2]

    if x > max_distance:
        max_distance = x
    else:
        if turns_since_max < 60 * 3:
            turns_since_max += 1
        else:
            turns_since_max = 0
            revert_turns = 3
            return 1 - result[0][0], 1 - result[0][1], 1 - result[0][2]

    return result[0][0], result[0][1], result[0][2]


def main():
    loop = asyncio.get_event_loop()
    mongo = setup_mongo(loop)
    db = mongo[config.DBNAME]
    collection = db[config.COLLECTION_NAME]

    try:
        loop.run_until_complete(train_and_save(collection))
    finally:
        mongo.close()


if __name__ == '__main__':
    main()
