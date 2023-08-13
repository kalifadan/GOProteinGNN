import os
import torch
from dataclasses import dataclass
import torch.nn as nn
from transformers import AutoConfig, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.file_utils import ModelOutput
from transformers.utils import logging
from typing import Union, Optional, Tuple
from src.knowledge_bert import KnowledgeBertModel

logger = logging.get_logger('pretrain_log')

PROTEIN_CONFIG_NAME = "config.json"
PROTEIN_MODEL_STATE_DICT_NAME = 'pytorch_model.bin'


class GOProteinGCNConfig:
    """
    contains configs for the GOProteinGCN model
    """

    def __init__(self, **kwargs):
        self.use_desc = kwargs.pop('use_desc', True)
        self.num_relations = kwargs.pop('num_relations', None)
        self.num_go_terms = kwargs.pop('num_go_terms', None)
        self.num_proteins = kwargs.pop('num_proteins', None)

        self.gcn_include_relations = kwargs.pop('gcn_include_relations', None)
        self.knowledge_layers_step = kwargs.pop('knowledge_layers_step', None)

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

        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is
            passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.

        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is
            passed or when ``config.output_attentions=True``):
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
    encoder_attention: Optional[Tuple[torch.FloatTensor]] = None
    go_attention: Optional[Tuple[torch.FloatTensor]] = None
    pooler_output: Optional[torch.FloatTensor] = None
    pos_pfi_logits: Optional[torch.FloatTensor] = None
    neg_pfi_logits: Optional[torch.FloatTensor] = None


class GOProteinGCN(nn.Module):
    """
    Implementation of the GOProteinGCN model
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.encoder_config = config.protein_model_config

        # Protein Knowledge Encoder
        self.encoder = KnowledgeBertModel(self.encoder_config, add_pooling_layer=False)

        # MLM head
        self.mlm_cls = BertOnlyMLMHead(self.encoder_config)

    def forward(self,
                protein_inputs: Tuple = None,
                pos_relation_inputs: Union[torch.Tensor, Tuple] = None,
                pos_go_tail_inputs: Union[torch.Tensor, Tuple] = None,
                neg_relation_inputs: Union[torch.Tensor, Tuple] = None,
                neg_go_tail_inputs: Union[torch.Tensor, Tuple] = None,
                use_pfi: bool = True,
                output_attentions: bool = False
                ):

        protein_input_ids, protein_attention_mask, protein_token_type_ids = protein_inputs

        # Protein Encoder
        protein_outputs = self.encoder(
            input_ids=protein_input_ids,
            attention_mask=protein_attention_mask,
            token_type_ids=protein_token_type_ids,
            input_relations=pos_relation_inputs,
            input_go_tails=pos_go_tail_inputs,
            output_hidden_states=True,
            return_dict=True,
            output_attentions=output_attentions
        )
        protein_seq_embed = protein_outputs[0]     # Shape: [batch size, seq len, embedding space]

        # Apply MLM head
        mlm_prediction_scores = self.mlm_cls(protein_seq_embed)

        return MaskedLMAndPFIOutput(
            mlm_loss=None,
            mlm_logits=mlm_prediction_scores,
            hidden_states=None,
            encoder_attention=None,
            go_attention=None,
            pooler_output=None,
            pos_pfi_logits=None,
            neg_pfi_logits=None
        )

    def save_pretrained(self, save_directory: os.PathLike, state_dict: Optional[dict] = None, save_config: bool = True):
        encoder_save_directory = os.path.join(save_directory, 'encoder')
        self.encoder.save_pretrained(encoder_save_directory, save_config=save_config)
        logger.info(f'Encoder Model weights saved in {encoder_save_directory}')

    @classmethod
    def from_pretrained(
            cls,
            protein_model_path: os.PathLike,
            text_model_path: os.PathLike,
            knowledge_model_file_name: os.PathLike,
            model_args=None,
            training_args=None,
            **kwargs
    ):

        # Will feed the number of relations and entity
        num_relations = kwargs.pop('num_relations', None)
        num_go_terms = kwargs.pop('num_go_terms', None)
        num_proteins = kwargs.pop('num_proteins', None)

        kmae_config = GOProteinGCNConfig.from_json_file(protein_model_path)

        kmae_config.protein_model_config.num_relations = num_relations
        kmae_config.protein_model_config.num_go_terms = num_go_terms
        kmae_config.protein_model_config.num_proteins = num_proteins

        if training_args:
            kmae_config.protein_model_config.use_desc = training_args.use_desc
            kmae_config.protein_model_config.use_pfi = training_args.use_pfi

        if model_args:
            kmae_config.protein_model_config.go_encoder_cls = model_args.go_encoder_cls
            kmae_config.protein_model_config.protein_encoder_cls = model_args.protein_encoder_cls
            kmae_config.protein_model_config.gcn_include_relations = model_args.gcn_include_relations
            kmae_config.protein_model_config.knowledge_layers_step = model_args.knowledge_layers_step

        kmae_config.protein_model_config.text_model_path = text_model_path

        # Instantiate model - textbert model is initializing in this step
        kmae_model = cls(config=kmae_config)

        # load model
        if kmae_model.encoder_config.protein_encoder_cls == 'bert':
            # if encoder state dict exists load encoder
            if knowledge_model_file_name is not None and \
                    os.path.exists(os.path.join(knowledge_model_file_name, 'pytorch_model.bin')):
                logger.info(f'Loading Model from {knowledge_model_file_name}')
                print(f'Loading Model from {knowledge_model_file_name}')
                kmae_model.encoder = KnowledgeBertModel.from_pretrained(knowledge_model_file_name)

            # if encoder state dict does not exists (first time training)
            else:
                print(f'Initializing weights from Pretrained-BERT Model: {protein_model_path}')
                # Initialize weights from BertEncoder
                bert_encoder = BertModel.from_pretrained(protein_model_path)
                kmae_model.encoder.encoder.layer = bert_encoder.encoder.layer
                kmae_model.encoder.embeddings = bert_encoder.embeddings
        else:
            raise NotImplementedError("Currently only support bert as encoder")

        kmae_model.eval()

        return kmae_model


@dataclass
class GOProteinGCNLoss:
    """
     Perform forward propagation and return loss for protein function inference

    for pfi task (default don't use):
        pfi_weight: weight of protein function inference loss
        num_protein_go_neg_sample: number of negative samples per positive sample  
    """

    def __init__(self, pfi_weight=1.0, num_protein_go_neg_sample=1, mlm_lambda=1.0):
        self.pfi_weight = pfi_weight
        self.mlm_lambda = mlm_lambda
        self.num_protein_go_neg_sample = num_protein_go_neg_sample
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(
            self,
            model: GOProteinGCN,
            use_desc: bool = False,
            use_seq: bool = True,
            use_pfi: bool = True,
            protein_go_inputs=None,
            **kwargs
    ):
        # get protein inputs
        protein_mlm_input_ids = protein_go_inputs['protein_input_ids']
        protein_mlm_attention_mask = protein_go_inputs['protein_attention_mask']
        protein_mlm_token_type_ids = protein_go_inputs['protein_token_type_ids']
        protein_input = (protein_mlm_input_ids, protein_mlm_attention_mask, protein_mlm_token_type_ids)

        protein_mlm_labels = protein_go_inputs['protein_labels']

        # relation inputs
        relation_ids = protein_go_inputs['relation_ids']
        relation_attention_mask = protein_go_inputs['relation_attention_mask']
        relation_token_type_ids = protein_go_inputs['relation_token_type_ids']
        relation_inputs = (relation_ids, relation_attention_mask, relation_token_type_ids)

        # positive inputs
        positive = protein_go_inputs['positive']

        # get tail inputs
        positive_tail_input_ids = positive['tail_input_ids']
        positive_tail_attention_mask = positive['tail_attention_mask']
        positive_tail_token_type_ids = positive['tail_token_type_ids']

        positive_go_tail_inputs = positive_tail_input_ids
        if use_desc:
            positive_go_tail_inputs = (
            positive_tail_input_ids, positive_tail_attention_mask, positive_tail_token_type_ids)

        # negative inputs
        negative_go_tail_inputs = None
        if use_pfi:
            negative = protein_go_inputs['negative']

            # get tail inputs
            negative_tail_input_ids = negative['tail_input_ids']
            negative_tail_attention_mask = negative['tail_attention_mask']
            negative_tail_token_type_ids = negative['tail_token_type_ids']

            negative_go_tail_inputs = negative_tail_input_ids
            if use_desc:
                negative_go_tail_inputs = (
                negative_tail_input_ids, negative_tail_attention_mask, negative_tail_token_type_ids)

        model_output = model(protein_inputs=protein_input,
                             pos_relation_inputs=relation_inputs,
                             pos_go_tail_inputs=positive_go_tail_inputs,
                             neg_relation_inputs=relation_inputs,
                             neg_go_tail_inputs=negative_go_tail_inputs,
                             use_pfi=use_pfi
                             )

        # mlm loss
        mlm_logits = model_output.mlm_logits
        batch, seq_len, vocab_size = mlm_logits.size()
        mlm_loss = self.loss_fn(mlm_logits.view(-1, vocab_size), protein_mlm_labels.view(-1)) * self.mlm_lambda

        # pfi loss
        pos_pfi_loss = 0
        neg_pfi_loss = 0
        if use_pfi:
            pos_pfi_logits = model_output.pos_pfi_logits  # (batch,2)
            neg_pfi_logits = model_output.neg_pfi_logits

            pos_pfi_label = protein_go_inputs['pfi_pos'].repeat(pos_pfi_logits.size(0))
            neg_pfi_label = protein_go_inputs['pfi_neg'].repeat(neg_pfi_logits.size(0))

            pos_pfi_loss = self.loss_fn(pos_pfi_logits.view(-1, 2), pos_pfi_label.view(-1)) * self.pfi_weight
            neg_pfi_loss = self.loss_fn(neg_pfi_logits.view(-1, 2), neg_pfi_label.view(-1)) * self.pfi_weight

        return mlm_loss, pos_pfi_loss, neg_pfi_loss


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (:obj:`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


# performs pooling that do not considers pads efficiently, supports max,avg and summation
def pool(h, mask, type='max'):
    # h dim (batch,seq len, feat dim); mask dim(batch, seq len,1|feat dim)
    if type == 'max':
        h = h.masked_fill(mask, -1e12)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


def copy_layers(src_layers, dest_layers, layers_to_copy):
    layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
    assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())
