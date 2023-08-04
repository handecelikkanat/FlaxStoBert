from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


class StoBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BertModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a BERT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.

    Examples::

        >>> from transformers import BertModel, BertConfig

        >>> # Initializing a BERT bert-base-uncased style configuration
        >>> configuration = BertConfig()

        >>> # Initializing a model from the bert-base-uncased style configuration
        >>> model = BertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0, #Hande
        attention_probs_dropout_prob=0, #Hande
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=0, #Hande #TODO: ASK TRUNG
        pad_token_id=0,
        gradient_checkpointing=False,
        position_embedding_type="absolute",
        use_cache=True,
        download_path="/users/zosaelai/cvbert_data/downloads",
        cache_path="/users/zosaelai/cvbert_data/cache",
        dataset="mnli-m",
        classifier_dropout=0, #Hande
        train_batch_size=16,
        eval_batch_size=16,
        num_labels=3,
        # Elaine: the next parameters are for Stochastic Bert
        stochastic=True,
        prior_mean=1.0,
        prior_std=0.5,
        rank=2,
        n_components=4,
        posterior_mean_init=(1.0, 0.5), #Trung: 1.0, 0.05
        posterior_std_init=(0.05, 0.02), #Trung: 0.0, 0.5
        gamma=1.0,
        kl_type='mean',
        entropy_type='conditional',
        mode='in',
        det_params={
            'lr': 5e-5, 'weight_decay': 5e-4
        },
        sto_params={
            'lr': 5e-5, 'weight_decay': 0.0
        },
        # sgd_params = {
        #     'momentum': 0.9,
        #     'dampening': 0.0,
        #     'nesterov': True
        # },
        kl_weight = {
            'kl_min': 0.0,
            'kl_max': 1.0,
            'last_iter': 5
        },
        num_train_samples=4,
        num_test_samples=2,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.download_path = download_path
        self.cache_path = cache_path
        self.classifier_dropout = classifier_dropout
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_labels = num_labels
        # Elaine: the following parameters are for Stochastic Bert
        self.stochastic = stochastic
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.n_components = n_components
        self.rank = rank
        self.posterior_mean_init = posterior_mean_init
        self.posterior_std_init = posterior_std_init
        self.gamma = gamma
        self.kl_type = kl_type
        self.entropy_type = entropy_type
        self.mode = mode
        self.det_params = det_params
        self.sto_params = sto_params
        # self.sgd_params = sgd_params
        self.kl_weight = kl_weight
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

