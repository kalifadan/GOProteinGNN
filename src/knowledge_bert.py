import os
from typing import Union
import torch
from dataclasses import dataclass
import torch.nn as nn
from transformers.models.bert.modeling_bert import BaseModelOutputWithPastAndCrossAttentions, BertEmbeddings
from transformers.file_utils import ModelOutput
from transformers.utils import logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from typing import Optional, Tuple, List
from torch_geometric.nn import GCNConv, RGCNConv
from transformers import AutoConfig, BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertPooler, BertLayer

logger = logging.get_logger('pretrain_log')

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

PROTEIN_CONFIG_NAME = "config.json"
PROTEIN_MODEL_STATE_DICT_NAME = 'pytorch_model.bin'


class KMAEConfig:
    """
    contains configs for the protein model
    """

    def __init__(self, **kwargs):
        self.use_desc = kwargs.pop('use_desc', True)
        self.num_relations = kwargs.pop('num_relations', None)
        self.num_go_terms = kwargs.pop('num_go_terms', None)
        self.num_proteins = kwargs.pop('num_proteins', None)

        self.protein_encoder_cls = kwargs.pop('protein_encoder_cls', None)
        self.go_encoder_cls = kwargs.pop('go_encoder_cls', None)

        self.protein_model_config = None

    def save_to_json_file(self, encoder_save_directory: os.PathLike):
        os.makedirs(encoder_save_directory, exist_ok=True)
        self.protein_model_config.save_pretrained(encoder_save_directory)
        logger.info(f'Encoder Configuration saved in {encoder_save_directory}')

    @classmethod
    def from_json_file(cls, encoder_config_path: os.PathLike):
        config = cls()
        config.protein_model_config = AutoConfig.from_pretrained(encoder_config_path)
        return config


@dataclass
class MaskedLMOutput(ModelOutput):
    """
    Base class for masked language models outputs.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Masked language modeling (MLM) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooler_output: Optional[torch.FloatTensor] = None


@dataclass
class MaskedLMAndPFIOutput(ModelOutput):
    mlm_loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pooler_output: Optional[torch.FloatTensor] = None
    pos_pfi_logits: Optional[torch.FloatTensor] = None
    neg_pfi_logits: Optional[torch.FloatTensor] = None


class KnowledgeGOProteinGCN(nn.Module):
    """
    Implementation of the full KnowledgeGOProteinGCN
    """
    def __init__(self, model_config=None):
        super().__init__()
        self.config = model_config

        # Text-Bert for relation and GO feature extraction, all param.requires_grad = False
        textbert_config = AutoConfig.from_pretrained(self.config.text_model_path)
        self.textbert = BertModel.from_pretrained(self.config.text_model_path, output_hidden_states=True)
        for param in self.textbert.parameters():
            param.requires_grad = False

        # linear layer to project features into the same dimension
        self.go_project = nn.Linear(textbert_config.hidden_size, self.config.hidden_size)

        # GCN
        in_channels = self.config.hidden_size  # embedding size
        out_channels = self.config.hidden_size  # embedding size

        # Use RGCN in case of gcn_include_relations to add edge_type (relation) to graph
        if self.config.gcn_include_relations:
            self.rgcn = RGCNConv(in_channels, out_channels, self.config.num_relations)
        else:
            self.gcn = GCNConv(in_channels, out_channels)

    def compute_text_embeddings(self, inputs: torch.Tensor, projection: nn.Linear):
        # feature extraction
        input_ids, attention_mask, token_type_ids = inputs
        go_relation_mask = (attention_mask == 0).all(dim=-1)

        # Reshape input tensors for batch processing and perform feature extraction for all GO tails together
        out = self.textbert(input_ids[~go_relation_mask],
                            attention_mask=attention_mask[~go_relation_mask],
                            token_type_ids=token_type_ids[~go_relation_mask],
                            output_hidden_states=True,
                            return_dict=True)  # Shape: (batch_size * num_go_tails, max_seq_length, feat_dim)

        # Extract the last hidden state of each GO tail
        out = out.hidden_states[0]
        feat = out[:, 0, :]   # Shape: (num_go_tails, feat_dim)

        # Apply go_project (projection layer) to each GO tail separately
        feat_matrix = projection(feat)  # Shape: (num_go_tails, hidden_dim)
        return feat_matrix, go_relation_mask

    def forward(self, relation_inputs, go_inputs, inputs_embeds):
        batch, protein_embed_size = inputs_embeds.size()
        device = inputs_embeds.device

        # GO feature extraction
        go_input_ids, go_attention_mask, go_token_type_ids = go_inputs
        go_feat_matrix, go_relation_mask = self.compute_text_embeddings(go_inputs, self.go_project)

        # Create node embeddings
        prot_seq_embed = inputs_embeds

        batch_size, max_relations_per_protein, max_seq_length = go_input_ids.size()
        node_embeddings = torch.cat([prot_seq_embed, go_feat_matrix.reshape(-1, protein_embed_size)], dim=0)

        # Define the source and target nodes for each edge
        num_proteins = batch_size
        num_relations_per_protein = torch.sum(~go_relation_mask, dim=1)

        # Create a mask to only select valid nodes based on the number of relations per protein
        valid_nodes_mask = torch.arange(max_relations_per_protein).unsqueeze(0).to(device) < num_relations_per_protein.unsqueeze(1).to(device)

        # Create the source nodes
        source_nodes = torch.arange(num_proteins).view(-1, 1).repeat(1, max_relations_per_protein).to(device)
        source_nodes = source_nodes[valid_nodes_mask]

        # Create the final target nodes
        target_nodes = num_proteins + torch.arange(valid_nodes_mask.flatten().sum()).to(device)

        # Combine the source and target nodes to create the edge_index tensor
        edge_index = torch.cat([source_nodes.flatten().unsqueeze(0), target_nodes.flatten().unsqueeze(0)], dim=0).to(device)

        # Apply RGCN with relation as edge_types or GCN if `gcn_include_relations` is false
        if self.config.gcn_include_relations:
            output = self.rgcn(node_embeddings, edge_index, relation_inputs)
        else:   # Apply GCN to protein sequence embeddings
            output = self.gcn(node_embeddings, edge_index)

        # Convert output to be match to the input dtype
        output = output.to(node_embeddings.dtype)

        # Split the GCN output back into protein and GO tail embeddings - now we use only the protein embeddings
        prot_seq_gcn_embed = output[:batch_size]  # Shape: (batch_size, hidden_dim)

        return prot_seq_gcn_embed


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

        # Initialize the Knowledge model
        self.knowledge_gcn = KnowledgeGOProteinGCN(config)
        self.knowledge_layers_step = config.knowledge_layers_step

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_relations: torch.Tensor,
        input_go_tails: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # Add CLS knowledge embeddings every few layers and in the last layer
            if (i + 1) % self.knowledge_layers_step == 0 or (i + 1) == len(self.layer):
                hidden_states = self.add_knowledge_cls_embeddings(hidden_states, input_relations, input_go_tails)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

    def add_knowledge_cls_embeddings(self, hidden_states, input_relations, input_go_tails):
        cls_embedding = hidden_states[:, 0:1, :].squeeze()    # Extract CLS embeddings
        knowledge_cls_embeddings = self.knowledge_gcn(input_relations, input_go_tails, cls_embedding)
        hidden_states = torch.cat([knowledge_cls_embeddings.unsqueeze(1), hidden_states[:, 1:, :]], dim=1)
        return hidden_states


class KnowledgeBertModel(BertPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        input_relations: Optional[torch.Tensor] = None,
        input_go_tails: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if input_relations is None:
            raise ValueError("You have to specify input_relations")
        else:   # convert relations to be one-dim tensor
            input_relations = input_relations[0]
            input_relations = torch.cat(input_relations)

        if input_go_tails is None:
            raise ValueError("You have to specify input_go_tails")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            input_relations=input_relations,
            input_go_tails=input_go_tails,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
