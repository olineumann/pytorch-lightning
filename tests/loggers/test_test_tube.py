import pickle

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from tests.base import LightningTestModel


def test_testtube_logger(tmpdir):
    """Verify that basic functionality of test tube logger works."""
    tutils.reset_seed()
    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)

    logger = tutils.get_default_testtube_logger(tmpdir, False)

    assert logger.name == 'lightning_logs'

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )
    result = trainer.fit(model)

    assert result == 1, 'Training failed'


def test_testtube_pickle(tmpdir):
    """Verify that pickling a trainer containing a test tube logger works."""
    tutils.reset_seed()

    hparams = tutils.get_default_hparams()

    logger = tutils.get_default_testtube_logger(tmpdir, False)
    logger.log_hyperparams(hparams)
    logger.save()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        train_percent_check=0.05,
        logger=logger
    )
    pkl_bytes = pickle.dumps(trainer)
    trainer2 = pickle.loads(pkl_bytes)
    trainer2.logger.log_metrics({'acc': 1.0})
