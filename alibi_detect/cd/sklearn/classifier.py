import logging
import numpy as np
from functools import partial
from typing import Callable, Dict, Optional, Tuple, List
from sklearn.base import clone, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from alibi_detect.cd.base import BaseClassifierDrift
from alibi_detect.utils.frameworks import has_alibi

if has_alibi:
    from alibi.explainers import KernelShap, TreeShap

logger = logging.getLogger(__name__)


class ClassifierDriftSklearn(BaseClassifierDrift):
    def __init__(
            self,
            x_ref: np.ndarray,
            model: ClassifierMixin,
            p_val: float = .05,
            preprocess_x_ref: bool = True,
            update_x_ref: Optional[Dict[str, int]] = None,
            preprocess_fn: Optional[Callable] = None,
            preds_type: str = 'probs',
            binarize_preds: bool = False,
            train_size: Optional[float] = .75,
            n_folds: Optional[int] = None,
            retrain_from_scratch: bool = True,
            seed: int = 0,
            use_calibration: bool = False,
            calibration_kwargs: Optional[dict] = None,
            use_oob: bool = False,
            use_shap: bool = False,
            shap_kwargs: Optional[dict] = None,
            data_type: Optional[str] = None,
            model_type: Optional[str] = None,
    ) -> None:
        """
        Classifier-based drift detector. The classifier is trained on a fraction of the combined
        reference and test data and drift is detected on the remaining data. To use all the data
        to detect drift, a stratified cross-validation scheme can be chosen.

        Parameters
        ----------
        x_ref
            Data used as reference distribution.
        model
            Sklearn classification model used for drift detection.
        p_val
            p-value used for the significance of the test.
        preprocess_x_ref
            Whether to already preprocess and store the reference data.
        update_x_ref
            Reference data can optionally be updated to the last n instances seen by the detector
            or via reservoir sampling with size n. For the former, the parameter equals {'last': n} while
            for reservoir sampling {'reservoir_sampling': n} is passed.
        preprocess_fn
            Function to preprocess the data before computing the data drift metrics.
        preds_type
            Whether the model outputs 'probs' or 'scores'.
        binarize_preds
            Whether to test for discrepancy on soft (e.g. probs/scores) model predictions directly
            with a K-S test or binarise to 0-1 prediction errors and apply a binomial test.
        train_size
            Optional fraction (float between 0 and 1) of the dataset used to train the classifier.
            The drift is detected on `1 - train_size`. Cannot be used in combination with `n_folds`.
        n_folds
            Optional number of stratified folds used for training. The model preds are then calculated
            on all the out-of-fold predictions. This allows to leverage all the reference and test data
            for drift detection at the expense of longer computation. If both `train_size` and `n_folds`
            are specified, `n_folds` is prioritized.
        retrain_from_scratch
            Whether the classifier should be retrained from scratch for each set of test data or whether
            it should instead continue training from where it left off on the previous set.
        seed
            Optional random seed for fold selection.
        use_calibration
            Whether to use calibration. Whether to use calibration. Calibration can be used on top of any model.
        calibration_kwargs
            Optional additional kwargs for calibration.
            See https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html
            for more details.
        use_oob
            Whether to use oob predictions. Supported only for RandomForestClassifier.
        use_shap
            Whether to use 'alibi' Shap explainer.
        shap_kwargs
            Kwargs for 'alibi' Shap explainer `fit` parameters. Only relevant for 'sklearn' backend.
        data_type
            Optionally specify the data type (tabular, image or time-series). Added to metadata.
        model_type
            Optionally specify the model type (tree-based). Added to metadata and used to chose the appropriate
            Shap explainer (i.e., if tree-based then TreeShap will be used for explanations).
        """
        super().__init__(
            x_ref=x_ref,
            p_val=p_val,
            preprocess_x_ref=preprocess_x_ref,
            update_x_ref=update_x_ref,
            preprocess_fn=preprocess_fn,
            preds_type=preds_type,
            binarize_preds=binarize_preds,
            train_size=train_size,
            n_folds=n_folds,
            retrain_from_scratch=retrain_from_scratch,
            seed=seed,
            data_type=data_type,
        )

        if use_shap and preprocess_x_ref:
            self.x_ref_orig = x_ref  # x_ref can be preprocessed in the __init__ of the base class

        if preds_type not in ['probs', 'scores']:
            raise ValueError("'preds_type' should be 'probs' or 'scores'")

        self.meta.update({'backend': 'sklearn'})
        self.original_model = model

        # calibration
        self.use_calibration = use_calibration
        self.calibration_kwargs = dict() if (calibration_kwargs is None) else calibration_kwargs

        # random forest oob flag
        self.use_oob = use_oob

        # shap explainer params
        self.use_shap = use_shap
        self.shap_kwargs = shap_kwargs if (shap_kwargs is not None) else dict()
        self.model_type = model_type

        # clone the original model and define explainer
        self.model = self._clone_model()

    def _create_explainer(self):
        if not self.use_shap:
            return None

        # use TreeShap
        if self.model_type == 'tree-based':
            # TODO: check the following statement
            # Can we do the same trick as in KernelShap by passing the raw dataset and modify the predict
            # function to apply the preprocessing internally?
            # This would be helpful when dealing with one-hot encodings, since we can aggregate all the values
            # for a categorical feature for free (see example in alibi) (i.e. no need for regrouping).
            # I believe that this might not possible since TreeShap needs the structure of the trees which are built
            # for preprocessed data. One should look into how the tree traversing is implemented for each instance
            # and check if there is a possibility to preprocess the data instance right before the tree traversing.
            # For now just throw an erro
            if isinstance(self.preprocess_fn, Callable):
                raise ValueError("`model_type='tree-based' cannot be used for ad-hoc data preprocessing. "
                                 "Preprocess the data before and set `preprocess_fn=None`.'")

            return TreeShap(predictor=self.model,
                            model_output=self.shap_kwargs.get('model_output', 'raw'),
                            feature_names=self.shap_kwargs.get('feature_names', None),
                            categorical_names=self.shap_kwargs.get('categorical_names', None),
                            task='classification',
                            seed=self.meta['seed'])

        # otherwise use KernelShap
        if self.preds_type == 'probs':
            predictor = (lambda x: self.model.predict_proba(self.preprocess_fn(x))) \
                if isinstance(self.preprocess_fn, Callable) else self.model.predict_proba
        else:
            predictor = (lambda x: self.model.decision_function(self.preprocess_fn(x))) \
                if isinstance(self.preprocess_fn, Callable) else self.model.decision_function

        return KernelShap(predictor=predictor,
                          link=self.shap_kwargs.get('link', 'identity'),
                          feature_names=self.shap_kwargs.get('feature_names', None),
                          categorical_names=self.shap_kwargs.get('categorical_names', None),
                          task='classification',
                          seed=self.meta['seed'],
                          distributed_opts=self.shap_kwargs.get('distributed_opts', None))

    def _clone_model(self):
        model = clone(self.original_model)

        # equivalence between `retrain_from_scratch` and `warm_start`
        if not self.retrain_from_scratch:
            if hasattr(model, 'warm_start'):
                model.warm_start = True
                logger.warning('`retrain_from_scratch=False` sets automatically the parameter `warm_start=True` '
                               'for the given classifier. Please consult the documentation to ensure that the '
                               '`warm_start=True` is applicable in the current context (i.e., for tree-based '
                               'models such as RandomForest, setting `warm_start=True` is not applicable since the '
                               'fit function expects the same dataset and an update/increase in the number of '
                               'estimators - previous fitted estimators will be kept frozen while the new ones '
                               'will be fitted).')
            else:
                logger.warning('Current classifier does not support `warm_start`. The model will be retrained '
                               'from scratch every iteration.')
        else:
            if hasattr(model, 'warm_start'):
                model.warm_start = False
                logger.warning('`retrain_from_scratch=True` sets automatically the parameter `warm_start=False`.')

        # oob checks
        if self.use_oob:
            if not isinstance(model, RandomForestClassifier):
                raise ValueError(f'OOB supported only for RandomForestClassifier. '
                                 f'Received a model of type {model.__class__.__name__}')

            if self.use_shap:
                self.use_shap = False
                logger.warning('`use_shap=True` cannot be used when `use_oob=True`. Setting `use_shap=False`.')

            if self.use_calibration:
                self.use_calibration = False
                logger.warning('Calibration cannot be user when `use_oob=True`. Setting `use_calibration=False`.')

            model.oob_score = True
            model.bootstrap = True
            logger.warning('`use_oob=True` sets automatically the parameter `boostrap=True` and `oob_score=True`.')
            logger.warning('`train_size` and `n_folds` are ignored when `use_oob=True`.')
        else:
            if isinstance(model, RandomForestClassifier):
                model.oob_score = False
                logger.warning('`use_oob=False` sets automatically the parameter `oob_score=False`')

        # preds_type checks
        if self.preds_type == 'probs':
            # calibrate the model if user specified.
            if self.use_calibration:
                logger.warning('Using calibration to obtain the prediction probabilities.')
                model = CalibratedClassifierCV(base_estimator=model, **self.calibration_kwargs)

            # check if shap can be used
            if self.use_shap and (not hasattr(model, 'predict_proba')):
                raise ValueError("Shap explainer cannot be used when `preds_type='probs'` and the classifier "
                                 "does not support `predict_proba`.")

            # if the binarize_preds=True, we don't really need the probabilities as in test_probs will be rounded
            # to the closest integer (i.e., to 0 or 1) according to the predicted probability. Thus, we can define
            # a hard label predict_proba based on the predict method
            if self.binarize_preds and (not hasattr(model, 'predict_proba')):
                if not hasattr(model, 'predict'):
                    raise AttributeError('Trying to use a model which does not support `predict`.')

                def predict_proba(self, X):
                    return np.eye(2)[self.predict(X).astype(np.int32)]

                # add predict_proba method
                model.predict_proba = partial(predict_proba, model)

            # at this point the model does not have any predict_proba, thus the test can not be performed.
            if not hasattr(model, 'predict_proba'):
                raise AttributeError("Trying to use a model which does not support `predict_proba` with "
                                     "`preds_type='probs'`. Set (`use_calibration=True`, `calibration_kwargs`) or "
                                     "(`binarize_preds=True`).")

        else:
            if self.use_calibration:
                logger.warning("No calibration is performed when `preds_type='scores'`.")

            if self.binarize_preds:
                raise ValueError("`binarize_preds` must be `False` when `preds_type='scores'`.")

            if not hasattr(model, 'decision_function'):
                raise AttributeError("Trying to use a model which does not support `decision_function` with "
                                     "`preds_type='scores'`.")

            # need to put the scores in the format expected by test function, which requires to duplicate the
            # scores along axis=1
            def predict_proba(self, X):
                scores = self.decision_function(X).reshape(-1, 1)
                return np.tile(scores, reps=2)

            # add predict_proba method
            model.predict_proba = partial(predict_proba, model)

        return model

    def score(self, x: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Compute the out-of-fold drift metric such as the accuracy from a classifier
        trained to distinguish the reference data from the data to be tested.

        Parameters
        ----------
        x
            Batch of instances.

        Returns
        -------
        p-value, a notion of distance between the trained classifier's out-of-fold performance \
        and that which we'd expect under the null assumption of no drift, \
        and the out-of-fold classifier model prediction probabilities on the reference and test data

        """
        if self.use_oob and isinstance(self.model, RandomForestClassifier):
            return self._score_rf(x)

        return self._score(x)

    def _compute_shap(self, x_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
        # create shap exaplainer
        # TODO: move this into constructor?
        shap_explainer = self._create_explainer()

        # fit shap explainer
        shap_fit_kwargs = {'background_data': x_tr}
        shap_fit_kwargs.update(self.shap_kwargs)  # this allows background data to be overwritten by None
        shap_explainer.fit(**shap_fit_kwargs)

        # select test data
        n_samples = shap_fit_kwargs.get('n_samples', x_te.shape[0])
        x_te = x_te[:n_samples]

        # explain test instances
        shap_explain_kwargs = {'X': x_te}
        shap_explain_kwargs.update(self.shap_kwargs)  # allow to pass some other argument to the `explain` method
        return shap_explainer.explain(**shap_explain_kwargs).shap_values[0]

    def _agregate_shap(self, shap_oof_list: List[np.ndarray]) -> np.ndarray:
        if len(shap_oof_list) == 0:
            return np.array([])

        shap_oof = np.concatenate(shap_oof_list, axis=0)
        shap_oof = np.mean(np.abs(shap_oof), axis=0)
        return shap_oof

    def _score(self, x: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # data used for shap explanation. note that is raw data (not preprocessed)
        if self.use_shap:
            x_ref_shap = self.x_ref_orig if hasattr(self, 'x_ref_orig') else self.x_ref
            x_shap = np.concatenate([x_ref_shap, x])

        x_ref, x = self.preprocess(x)
        n_ref, n_cur = len(x_ref), len(x)
        x, y, splits = self.get_splits(x_ref, x)

        # iterate over folds: train a new model for each fold and make out-of-fold (oof) predictions
        probs_oof_list, idx_oof_list = [], []
        shap_oof_list = []

        for idx_tr, idx_te in splits:
            y_tr = y[idx_tr]

            if isinstance(x, np.ndarray):
                x_tr, x_te = x[idx_tr], x[idx_te]
            elif isinstance(x, list):
                x_tr, x_te = [x[_] for _ in idx_tr], [x[_] for _ in idx_te]
            else:
                raise TypeError(f'x needs to be of type np.ndarray or list and not {type(x)}.')

            # fit the model and extract probs
            self.model.fit(x_tr, y_tr)
            probs = self.model.predict_proba(x_te)
            probs_oof_list.append(probs)
            idx_oof_list.append(idx_te)

            # explain the model
            if self.use_shap:
                shap_oof_list.append(self._compute_shap(x_tr=x_shap[idx_tr], x_te=x_shap[idx_te]))

        # concatenate oof probs and indices
        probs_oof = np.concatenate(probs_oof_list, axis=0)
        idx_oof = np.concatenate(idx_oof_list, axis=0)

        # extract oof corresponding outputs
        y_oof = y[idx_oof]

        # compute p-values and distance
        p_val, dist = self.test_probs(y_oof, probs_oof, n_ref, n_cur)

        # sort the oof probs to correspond to ref and curr
        probs_sort = probs_oof[np.argsort(idx_oof)]

        # compute global shap explanation
        shap_oof = self._agregate_shap(shap_oof_list) if self.use_shap else None
        return p_val, dist, probs_sort[:n_ref, 1], probs_sort[n_ref:, 1], shap_oof

    def _score_rf(self, x: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        x_ref, x = self.preprocess(x)
        x, y, _ = self.get_splits(x_ref, x, return_splits=False)
        self.model.fit(x, y)

        # it is possible that some inputs do not have OOB scores. This is probably means that too few trees were
        # used to compute any reliable estimates.
        # TODO: Shall we raise an error or keep going by selecting only not NaN?
        index_oob = np.where(np.all(~np.isnan(self.model.oob_decision_function_), axis=1))[0]
        probs_oob = self.model.oob_decision_function_[index_oob]
        y_oob = y[index_oob]

        # comparison due to ordering in get_split (i.e, x = [x_ref, x])
        n_ref = np.sum(index_oob < len(x_ref)).item()
        n_cur = np.sum(index_oob >= len(x_ref)).item()
        p_val, dist = self.test_probs(y_oob, probs_oob, n_ref, n_cur)
        return p_val, dist, probs_oob[:n_ref, 1], probs_oob[n_ref:, 1]
