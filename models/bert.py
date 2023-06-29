from typing import Optional, Type
import jax
import haiku as hk
import jax.numpy as jnp

class BertEmbeddings(hk.Module):
    def __init__(self, config, name=None):
        super().__init__(name)
        self.word_embeddings = hk.Embed(vocab_size=config.vocab_size, embed_dim=config.hidden_size)
        self.position_embeddings = hk.Embed(vocab_size=config.max_position_embeddings, embed_dim=config.hidden_size)
        self.token_type_embeddings = hk.Embed(vocab_size=config.type_vocab_size, embed_dim=config.hidden_size)

        self.layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, eps=config.layer_norm_eps)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.max_position_embeddings = config.max_position_embeddings

    def __call__(self, input_ids = None, token_type_ids = None, position_ids = None, inputs_embeds = None, past_key_values_length = 0, is_training=False):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]
        
        seq_length = input_shape[1] # [batch_size, seq_length, ...]

        if position_ids is None:
            position_ids = jnp.arange(0, self.max_position_embeddings)[jnp.newaxis, :]
            position_ids = position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = jnp.zeros(input_shape, dtype=jnp.int32)
        
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        embeddings = self.layer_norm(embeddings)
        if is_training:
            embeddings = hk.dropout(hk.next_rng_key(), self.hidden_dropout_prob, embeddings)
        return embeddings