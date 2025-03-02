import os
import yaml
from ultralytics import YOLO
from typing import List
from pymilvus import (
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import pymilvus
from loguru import logger
from pymilvus import connections
from sklearn.preprocessing import normalize

from app.settings import Settings

settings = Settings()

face_model_path = settings.yolo.embedding_weight

D_TYPE = {
    "BOOL": DataType.BOOL,
    "INT8": DataType.INT8,
    "INT16": DataType.INT16,
    "INT32": DataType.INT32,
    "INT64": DataType.INT64,
    "VARCHAR": DataType.VARCHAR,
    "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
}


class EmbeddingModel:
    def __init__(self):
        # Load face recognition model (assuming classification-based YOLO for faces)
        if not os.path.exists(face_model_path):
            raise FileNotFoundError(
                f"Face recognition model not found: {face_model_path}"
            )
        self.face_embedding_model = YOLO(face_model_path)

    def __call__(self, face_img):
        embedding_face_image = self.face_embedding_model.embed(face_img)[0].numpy()
        return normalize(embedding_face_image.reshape(1, -1), norm="l2").flatten()


class Milvus:
    fmt = "\n=== {:30} ===\n"
    connections.connect(
        alias=settings.milvus.alias,
        host=settings.milvus.host,
        port=settings.milvus.port,
    )

    def __init__(self):
        config = self.__load_yaml__(settings.milvus.config_path)
        self.collection_name = config["COLLECTION_NAME"]

    def __load_yaml__(self, path):
        with open(path, "r") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    def __create_collection__(self):
        fields = [
            FieldSchema(
                name="item_id", dtype=DataType.INT64, is_primary=True, auto_id=False
            )
        ]

        config_table = self.cfg["TABLE_CONFIG"]
        for col in self.cfg["TABLE_CONFIG"]:
            if "DIM" in config_table[col]:
                fields.append(
                    FieldSchema(
                        name=config_table[col]["NAME"],
                        dtype=D_TYPE[config_table[col]["D_TYPE"]],
                        dim=config_table[col]["DIM"],
                    )
                )
            else:
                fields.append(
                    FieldSchema(
                        name=config_table[col]["NAME"],
                        dtype=D_TYPE[config_table[col]["D_TYPE"]],
                        max_length=config_table[col]["MAX_LEN"],
                    )
                )

        schema = CollectionSchema(
            fields, description=f"Vector DB Search for a `{self.collection_name}`."
        )

        logger.info(self.fmt.format(f"Create collection `{self.collection_name}`"))
        return Collection(self.collection_name, schema, consistency_level="Strong")

    def delete(self, collection_name=None):
        if collection_name is not None:
            collection_name = collection_name
        else:
            collection_name = self.collection_name

        logger.info(self.fmt.format(f"Delete collection {collection_name}..."))
        pymilvus.drop_collection(collection_name)

    def insert(self, data: List[List[str]]):
        if not self.check_available():
            milvus_table = self.__create_collection__()
        else:
            milvus_table = Collection(self.collection_name)
            milvus_table.release()

        # Create table
        logger.info(self.fmt.format("Start inserting entities..."))
        milvus_table.insert(data)
        milvus_table.flush()  # Remove in mem

    def create_index(self):
        milvus_table = Collection(self.collection_name)
        # Create index
        logger.info(self.fmt.format("Start Creating index IVF_FLAT"))
        milvus_table.create_index(
            self.cfg["INDEXING"]["FIELD_NAME"],
            index_params=self.cfg["INDEXING"]["PARAMS"],
        )

    def check_available(self):
        return utility.has_collection(self.collection_name)


class Search:
    def __init__(self, config):
        self.cfg = config
        self.collection = Collection(config["COLLECTION_NAME"])
        self.collection.load()

    def __call__(self, vector: List[float], limit: int = 10, offset: int = 0):
        try:
            self.cfg["SEARCH"]["PARAMS"]["data"] = [vector]
            self.cfg["SEARCH"]["PARAMS"]["limit"] = limit
            self.cfg["SEARCH"]["PARAMS"]["offset"] = offset
            self.cfg["SEARCH"]["PARAMS"]["expr"] = None

            results = self.collection.search(**self.cfg["SEARCH"]["PARAMS"])
            outputs = []
            for hits in results:
                for hit in hits:
                    outputs.append(
                        {
                            "id": hit.id,
                            "answer": hit.entity.get(
                                self.cfg["SEARCH"]["PARAMS"]["output_fields"][0]
                            ),
                            "distance": hit.distance,
                        }
                    )
            return outputs
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return []
