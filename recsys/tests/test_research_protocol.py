import hashlib
import unittest

from recsys.bundle import canonical_bytes
from recsys.research_protocol import ResearchError, load_protocol, order_hash, protocol_identity, split_user


class ProtocolTests(unittest.TestCase):
    def test_frozen_policy_and_split(self):
        protocol = load_protocol("ml01-v1")
        self.assertEqual(protocol["interaction_policy"]["positive_min_rating"], 8)
        self.assertEqual(protocol["comparison"]["epsilon_ndcg5"], .01)
        self.assertEqual(protocol["query"]["order_seeds"], [361, 362, 363])
        user = "u1_" + "a" * 64
        digest = hashlib.sha256(b"arb-ml-user-split-v1\0" + b"36\0" + user.encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") % 100
        expected = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
        self.assertEqual(split_user(user), expected)
        self.assertEqual(split_user(user), expected)
        self.assertNotEqual(order_hash(361, user, 1), order_hash(362, user, 1))
        self.assertTrue(protocol_identity().startswith("sha256:"))

    def test_strict_types_and_canonical(self):
        protocol = load_protocol()
        for change in (lambda p: p.update(schema_version=True),
                       lambda p: p.update(unexpected=True)):
            copy = __import__("json").loads(canonical_bytes(protocol))
            change(copy)
            with self.assertRaises(ResearchError):
                load_protocol(_protocol_bytes=canonical_bytes(copy))
        with self.assertRaises(ResearchError):
            load_protocol(_protocol_bytes=canonical_bytes(protocol).replace(b"0.01", b"NaN"))
        with self.assertRaises(ResearchError):
            load_protocol("unknown")
        with self.assertRaises(ResearchError):
            split_user("u1_" + "A" * 64)
