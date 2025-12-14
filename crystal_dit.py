import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp

# 从DiT导入必要的组件
from diffusion.models import modulate, DiTBlock, TimestepEmbedder

# Custom 1D positional encoding function
def get_1d_sincos_pos_embed(embed_dim, length):
    """Generate 1D sinusoidal positional embeddings"""
    assert embed_dim % 2 == 0, "Embedding dimension must be even"
    
    positions = np.arange(length)
    dim_t = np.arange(embed_dim // 2, dtype=np.float32)
    dim_t = 10000 ** (2 * (dim_t // 2) / embed_dim)
    
    pos_x = positions[:, np.newaxis] / dim_t
    pos_embed = np.zeros((length, embed_dim))
    
    pos_embed[:, 0::2] = np.sin(pos_x)
    pos_embed[:, 1::2] = np.cos(pos_x)
    
    return pos_embed

# 文本映射
class ProjectionHead(nn.Module):
    """
        将外部给定的文本特征映射到模型使用的低维空间
    """
    def __init__(self,embedding_dim,projection_dim=256,dropout=0.1):  #256,64
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CrystalEmbedder(nn.Module):
    """Embed crystal structure (lattice vectors and atom features) into hidden space"""
    
    def __init__(self, hidden_size, max_atoms):
        super().__init__()
        self.max_atoms = max_atoms
        
        # Lattice vector embedder
        self.lattice_embedder = nn.Linear(3, hidden_size)
        
        # Atom feature embedder (period, group, x, y, z)
        self.atom_embedder = nn.Linear(5, hidden_size)

        # text embedder
        # self.structure_embedder1 = ProjectionHead(embedding_dim=768)
        # self.properties_embedder1 = ProjectionHead(embedding_dim=768)

        # self.structure_embedder = nn.Linear(768, hidden_size)
        # self.properties_embedder = nn.Linear(768, hidden_size)
        # self.structure_embedder = ProjectionHead(embedding_dim=768, projection_dim=512)
        # self.properties_embedder = ProjectionHead(embedding_dim=768, projection_dim=512)
        self.text_embedder = ProjectionHead(embedding_dim=768, projection_dim=512)

        # self.structure_embedder = nn.ModuleList([ProjectionHead(embedding_dim=768),
        #                                         nn.Linear(768, hidden_size)])
        # self.properties_embedder = nn.ModuleList([ProjectionHead(embedding_dim=768),
        #                                         nn.Linear(768, hidden_size)])


        
        # Type embeddings to distinguish lattice and atom tokens
        self.lattice_type_embedding = nn.Parameter(torch.zeros(1, hidden_size))
        self.atom_type_embedding = nn.Parameter(torch.zeros(1, hidden_size))
        # self.struct_type_embedding = nn.Parameter(torch.zeros(1, hidden_size))
        # self.prop_type_embedding = nn.Parameter(torch.zeros(1, hidden_size))
        self.text_embedding = nn.Parameter(torch.zeros(1, hidden_size))
        
        # Positional embeddings for lattice vectors
        self.lattice_pos_embed = nn.Parameter(torch.zeros(1, 3, hidden_size), requires_grad=False)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.lattice_embedder.weight)
        nn.init.constant_(self.lattice_embedder.bias, 0)

        nn.init.xavier_uniform_(self.atom_embedder.weight)
        nn.init.constant_(self.atom_embedder.bias, 0)

        # nn.init.xavier_uniform_(self.structure_embedder1.weight)
        # nn.init.constant_(self.structure_embedder1.bias, 0)
        
        # nn.init.xavier_uniform_(self.properties_embedder1.weight)
        # nn.init.constant_(self.properties_embedder1.bias, 0)
        
        # nn.init.xavier_uniform_(self.structure_embedder.weight)
        # nn.init.constant_(self.structure_embedder.bias, 0)
        
        # nn.init.xavier_uniform_(self.properties_embedder.weight)
        # nn.init.constant_(self.properties_embedder.bias, 0)

        # Initialize lattice positional embeddings with sinusoidal encoding
        pos_embed = get_1d_sincos_pos_embed(self.lattice_pos_embed.shape[-1], 3)
        self.lattice_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        nn.init.normal_(self.lattice_type_embedding, std=0.02)
        nn.init.normal_(self.atom_type_embedding, std=0.02)
        # nn.init.normal_(self.struct_type_embedding, std=0.02)
        # nn.init.normal_(self.prop_type_embedding, std=0.02)
        nn.init.normal_(self.text_embedding, std=0.02)
    
    def forward(self, text_vectors, lattice_vectors, atom_features):
        """
        Args:
            lattice_vectors: [batch_size, 3, 3] - lattice vectors
            atom_features: [batch_size, max_atoms, 5] - atom features (period, group, x, y, z)
        Returns:
            lattice_emb: [batch_size, 3, hidden_size] - lattice embeddings
            atom_emb: [batch_size, max_atoms, hidden_size] - atom embeddings
        """
        # print(f'in crystalembedder before--------------------')
        # print(f'structure_emb: {structure_emb.shape}')
        # print(f'properties_emb: {properties_emb.shape}')
        # print(f'lattice_emb: {lattice_emb.shape}')
        # print(f'atom_emb: {atom_emb.shape}')
        # print(f'in crystalembedder end --------------------')

        lattice_emb = self.lattice_embedder(lattice_vectors)
        lattice_emb = lattice_emb + self.lattice_pos_embed + self.lattice_type_embedding
        
        atom_emb = self.atom_embedder(atom_features)
        atom_emb = atom_emb + self.atom_type_embedding

        # structure_emb = self.structure_embedder(structure_vectors)
        # structure_emb = structure_emb + self.struct_type_embedding

        # properties_emb = self.structure_embedder(properties_vectors)
        # properties_emb = properties_emb + self.prop_type_embedding

        text_emb = self.text_embedder(text_vectors)
        text_emb = text_emb + self.text_embedding

        # print(f'in crystalembedder--------------------')
        # print(f'structure_emb: {structure_emb.shape}') # structure_emb: torch.Size([256, 1, 512])
        # print(f'properties_emb: {properties_emb.shape}') # properties_emb: torch.Size([256, 1, 512])
        # print(f'lattice_emb: {lattice_emb.shape}') # lattice_emb: torch.Size([256, 3, 512])
        # print(f'atom_emb: {atom_emb.shape}') # atom_emb: torch.Size([256, 20, 512])
        # print(f'in crystalembedder end --------------------')

        
        return text_emb, lattice_emb, atom_emb


class SimpleDiTBlock(nn.Module):
    """Simple DiT block that concatenates atom and lattice features for self-attention"""
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, max_atoms=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_atoms = max_atoms
        self.dit_block = DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
        
    # todo: adding text 
    def forward(self, text_features, atom_features, lattice_features, c):
        """
        Args:
            atom_features: [batch_size, max_atoms, hidden_size]
            lattice_features: [batch_size, 3, hidden_size]
            c: [batch_size, hidden_size] - timestep conditioning
        Returns:
            Updated atom and lattice features
        """
        # Concatenate features
        combined_features = torch.cat([atom_features, lattice_features], dim=1)
        # print(f'combined_features before dit: {combined_features.shape}')

        # combined_features2 = torch.cat([atom_features, lattice_features, structure_features], dim=1)
        # print(f'combined_features 2: {combined_features2.shape}')
        # print(f'c: {c.shape}')

        
        # Apply DiT block
        combined_features = self.dit_block(x_q=combined_features, x_kv=text_features, c=c)
        # print(f'combined_features after dit: {combined_features.shape}' )
        
        # Split back
        atom_features_out = combined_features[:, :self.max_atoms, :]
        lattice_features_out = combined_features[:, self.max_atoms:, :]
        
        return atom_features_out, lattice_features_out


class FinalLayer(nn.Module):
    """Final layer for DiT model output"""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final_lattice = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_final_atom = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        self.lattice_linear = nn.Linear(hidden_size, 3, bias=True)
        self.atom_linear = nn.Linear(hidden_size, 5, bias=True)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )

    def forward(self, atom_features, lattice_features, c):
        shift_atom, scale_atom, shift_lattice, scale_lattice = self.adaLN_modulation(c).chunk(4, dim=1)
        
        atom_features = modulate(self.norm_final_atom(atom_features), shift_atom, scale_atom)
        lattice_features = modulate(self.norm_final_lattice(lattice_features), shift_lattice, scale_lattice)
        
        lattice_out = self.lattice_linear(lattice_features)
        atom_out = self.atom_linear(atom_features)
        
        return lattice_out, atom_out


class CrystalDiT(nn.Module):
    """Crystal DiT model using simple concatenated self-attention architecture"""
    
    def __init__(
        self,
        max_atoms=20,
        hidden_size=512,
        depth=18,
        num_heads=8,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.max_atoms = max_atoms
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        
        # self.structure_projection = ProjectionHead(embedding_dim=768)
        # self.properties_projection = ProjectionHead(embedding_dim=768)

        self.crystal_embedder = CrystalEmbedder(hidden_size, max_atoms)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.blocks = nn.ModuleList([
            SimpleDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, max_atoms=max_atoms)
            for _ in range(depth)
        ])
        
        self.final_layer = FinalLayer(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        # Initialize modulation layers to zero
        for block in self.blocks:
            nn.init.constant_(block.dit_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.dit_block.adaLN_modulation[-1].bias, 0)
        
        # Initialize final layer to zero
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.lattice_linear.weight, 0)
        nn.init.constant_(self.final_layer.lattice_linear.bias, 0)
        nn.init.constant_(self.final_layer.atom_linear.weight, 0)
        nn.init.constant_(self.final_layer.atom_linear.bias, 0)

    # todo: adding text embeddings as additional model inputs
    def forward(self, text_vectors, lattice_vectors, atom_features, t):
        """
        Args:
            lattice_vectors: [batch_size, 3, 3] - lattice vectors
            atom_features: [batch_size, max_atoms, 5] - atom features (period, group, x, y, z)
            t: [batch_size] - diffusion timesteps
        Returns:
            lattice_out: [batch_size, 3, 3] - predicted lattice noise
            atom_out: [batch_size, max_atoms, 5] - predicted atom noise
        """
        # structure_vectors = self.structure_projection(structure_vectors)
        # properties_vectors = self.properties_projection(properties_vectors)

        # print(f'in crystaldit ----------------')
        # print(f'structure_vectors: {structure_vectors.shape}') # structure_vectors: torch.Size([256, 1, 768])
        # print(f'properties_vectors: {properties_vectors.shape}') # properties_vectors: torch.Size([256, 1, 768])
        # print(f'in crystaldit end ----------------')


        text_emb, lattice_emb, atom_emb = self.crystal_embedder(
            text_vectors, lattice_vectors, atom_features
        )
        c = self.t_embedder(t)

        # print(f'in crystaldit ----------------')
        # print(f'structure_emb: {structure_emb.shape}') # torch.Size([256, 1, 512])
        # print(f'properties_emb: {properties_emb.shape}') # torch.Size([256, 1, 512])
        # print(f'lattice_emb: {lattice_emb.shape}') # torch.Size([256, 3, 512])
        # print(f'atom_emb: {atom_emb.shape}') # torch.Size([256, 20, 512])
        # print(f'in crystaldit end ----------------')
        
        # structure_features = structure_emb
        # properties_features = properties_emb
        text_features = text_emb
        atom_features = atom_emb
        lattice_features = lattice_emb
        
        # just use properties_features for all blocks
        for block in self.blocks:
            atom_features, lattice_features = block(text_features, atom_features, lattice_features, c)
            # atom_features, lattice_features = block(structure_features, properties_features, atom_features, lattice_features, c) # back

        #! 修改模型文本引导变为 FilmLayer ?   并且只使用 属性文本
        '''  odd-even blocks use different text features
        for i, block in enumerate(self.blocks):
            if i % 2 == 0:
                atom_features, lattice_features = block(structure_features, atom_features, lattice_features, c)
                # print(f'--------- use structure_features!!!!')
            else:
                atom_features, lattice_features = block(properties_features, atom_features, lattice_features, c)
                # print(f'--------- use properties_features!!!!')
        '''  

        # atom_features, lattice_features = block(structure_features, atom_features, lattice_features, c)
        lattice_out, atom_out = self.final_layer(atom_features, lattice_features, c)
        lattice_out = lattice_out.view(-1, 3, 3)
        
        return lattice_out, atom_out
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
