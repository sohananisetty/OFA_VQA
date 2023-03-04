from dataclasses import dataclass, field

@dataclass
class VQAArguments:
 """
Some custom parameters
 Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
 """
 max_seq_length : int  =  field ( metadata = { "help" : "Enter the maximum length" })
 patch_image_size : int  =  field ( metadata = { "help" : "Image Resolution" })
 label_smoothing : float  =  field ( metadata = { "help" : "label_smoothing" })
 data_folder : str  =  field ( metadata = { "help" : "data_folder" })
 pretrained : str  =  field ( metadata = { "help" : "Pre-training weight path" })
 freeze_encoder : bool  =  field ( metadata = { "help" : "Whether to freeze the weight of the encoder and only finetune the decoder" })
 freeze_word_embed: bool = field( metadata = { "help" : "Whether to freeze the weight of the word vector of the encoder, since the enocder of the OFA model shares the weight of the word vector with the decoder, freeze_encoder will freeze the word vector. When freeze_word_embed=False, the word vector will be trained together" })