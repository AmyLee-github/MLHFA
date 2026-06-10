import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, num_channels, num_heads):
        super(CrossAttention, self).__init__()
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads)
        self.output_conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)

    def forward(self, x1, x2):
        batch_size, num_channels, height, width = x1.size()

        # Generate queries, keys, and values
        queries = self.query_conv(x1).view(batch_size, num_channels, -1).permute(2, 0, 1)
        keys = self.key_conv(x2).view(batch_size, num_channels, -1).permute(2, 0, 1)
        values = self.value_conv(x2).view(batch_size, num_channels, -1).permute(2, 0, 1)

        # Compute multi-head cross-attention
        attn_output, _ = self.multihead_attn(queries, keys, values)
        attn_output = attn_output.permute(1, 2, 0).view(batch_size, num_channels, height, width)

        # Apply final convolution
        output = self.output_conv(attn_output)

        return output

# Example usage
if __name__ == "__main__":
    batch_size = 8
    num_channels = 64
    height = 32
    width = 32
    num_heads = 8

    x1 = torch.randn(batch_size, num_channels, height, width)
    x2 = torch.randn(batch_size, num_channels, height, width)

    model = CrossAttention(num_channels, num_heads)
    output = model(x1, x2)
    print(output.shape)
