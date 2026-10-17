"""Small train-only fixture with held-out users for native training checks."""

from pathlib import Path

from research_model_support import ModelFixture

from recsys.data.dataset import HEADER
from recsys.research.artifacts import StreamWriter
from recsys.research.protocol import load_protocol, split_user


class SeedTrainingFixture(ModelFixture):
    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        protocol = load_protocol()
        train = [user for user in self.users if split_user(user, protocol) == "train"][:5]
        validation = [user for user in self.users if split_user(user, protocol) == "validation"][:1]
        test = [user for user in self.users if split_user(user, protocol) == "test"][:1]
        if len(train) != 5 or len(validation) != 1 or len(test) != 1:
            raise AssertionError("fixture_user_split_insufficient")
        profiles: dict[str, dict[int, tuple[int, str]]] = {
            train[0]: {
                1: (10, "completed"),
                2: (8, "completed"),
                3: (2, "completed"),
                15: (10, "watching"),
            },
            train[1]: {1: (7, "completed"), 3: (9, "completed"), 4: (8, "completed")},
            train[2]: {2: (1, "completed"), 5: (10, "completed"), 6: (8, "completed")},
            train[3]: {7: (8, "completed")},
            train[4]: {
                1: (8, "completed"),
                2: (1, "completed"),
                3: (1, "completed"),
                4: (1, "completed"),
                5: (1, "completed"),
                6: (1, "completed"),
                7: (1, "completed"),
                8: (9, "completed"),
            },
            validation[0]: {9: (9, "completed"), 10: (8, "completed")},
            test[0]: {11: (10, "completed"), 12: (8, "completed")},
        }
        path = self.dataset / "interactions.tsv.gz"
        path.unlink()
        stream = StreamWriter(path)
        stream.write(HEADER)
        for user in sorted(profiles):
            for item, (rating, status) in sorted(profiles[user].items()):
                stream.write(f"{user}\t{item}\t{rating}\t{status}\n".encode("ascii"))
        stream.close()
        self.training_users = tuple(train)
        self.validation_user = validation[0]
        self.test_user = test[0]
