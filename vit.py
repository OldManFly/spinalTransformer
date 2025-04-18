import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import nibabel as nib

def pair(t):
    """Helper function to ensure that the input is a tuple of two elements."""
    return t if isinstance(t, tuple) else (t, t)

class feedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()  ##確保nn.Module有正確被初始化
        ### 按照順序定義網路的結構
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        print(f"Input to LayerNorm: {x.shape}")
        return self.net(x)  ##將輸入x傳入net中，並回傳結果

# Define the attention mechanism
class attention(nn.Module):
    def __init__(self, input_dim, embedding_head = 8, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head*embedding_head
        ## 判斷是否需要輸出投影層。如果 heads 為 1 且 dim_head 等於 dim，則不需要，因為注意力輸出的維度已經與輸入相同。
        project_out = not (input_dim == inner_dim and dropout == 0)  
        self.heads = embedding_head
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, input_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity() 


    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# Define the transformer model
class transformer(nn.Module):
    def __init__(self, input_dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                attention(input_dim, heads, dim_head, dropout),
                feedForward(input_dim, mlp_dim, dropout)
            ]))
        
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    

# Define the ViT model
class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, depth, num_classes, dim, heads, mlp_dim, pool='cls', channels=1, dim_head=64, dropout=0., emb_dropout=0., patch_depth=4):
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth_val = (*pair(image_patch_size), patch_depth)
        assert image_depth % patch_depth == 0 and image_width % patch_width == 0 and image_width % patch_width == 0, 'Image depth must be divisible by patch depth'

        num_patches = (image_depth // patch_depth) * (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_depth * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool must be either cls or mean'

        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_depth = patch_depth_val
        self.channels = channels
        self.dim = dim

        self.to_patch_embedding = nn.Sequential(
              nn.LayerNorm(patch_dim),  # LayerNorm input should be dim
              nn.Linear(patch_dim, dim),  # Linear input should be dim, output should be 512
              nn.LayerNorm(dim),
              nn.GELU(),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim) 
        )

        # 分割頭 (需要根據你的具體任務調整)
        self.upsample = nn.ConvTranspose3d(dim, dim, kernel_size=4, stride=2)  # 修改這裡
        self.final_conv = nn.Conv3d(dim, num_classes, kernel_size=1)  # 修改這裡

    def forward(self, images):
        b, h, w, d = images.shape
        ph, pw, pd = self.patch_height, self.patch_width, self.patch_depth

        # 手動實現 Rearrange
        x = images.reshape(b, h // ph, ph, w // pw, pw, d // pd, pd)
        x = x.permute(0, 2, 4, 6, 1, 3, 5)
        x = x.reshape(b, (h // ph) * (w // pw) * (d // pd), 1 * ph * pw * pd)

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)

        x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Add spatial dims
        x = self.upsample(x)
        x = self.final_conv(x)  # Output shape: [b, num_classes, H, W, D]
        return x
    
##### Testing part  ######


if __name__ =='__main__':
    niiImage = nib.load('/media/oldman/OldmanDoc/Document/NYCU/transformer/MRSpineSeg_Challenge_SMU/train/MR/Case8.nii.gz')
    image = niiImage.get_fdata()
    image_size = image.shape
    print(image_size)
    image_patch_size = (88, 88) # 在高度和寬度方向的 patch 大小
    patch_depth = 4 # 在深度方向的 patch 大小
    num_classes = 10 # 您的分類類別數量
    dim = 32
    depth = 15
    heads = 8
    mlp_dim = 128
    dropout = 0.1
    emb_dropout = 0.1

    model = ViT(
        image_size=image_size,
        image_patch_size=image_patch_size,
        patch_depth=patch_depth,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout,
        channels=1 # 根據您的影像通道數調整
    )

    dummy_input = torch.randn(4, 1, 880, 880, 15)
    output = model(dummy_input)
    print(model)
    # print(output.shape) # 預期輸出 shape 為 (batch_size, num_classes)
