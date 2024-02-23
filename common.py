################ CA ##################

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

############### Ghostnet V2 ######################
class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)
      
class GhostBottleneckV2(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            DFC(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        x1 = self.shortcut(x)
        x2 = self.conv(x)

        return x1 + x2

class DFC(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):  # ch_in, ch_out, kernel, stride, groups
        super(DFC, self).__init__()
        self.gate_fn = nn.Sigmoid()
        self.c2 = c2

        self.short_conv = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.conv = GhostConv(c1, c2, 1, 1, act=False)  # pw-linear

    def forward(self, x):
        res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2)) 
        out = self.conv(x)

        out1 = out[:, :self.c2, :, :]  # tensor:(2, 16, 64, 64)
        shp1 = out1.shape

        out2 = F.interpolate(self.gate_fn(res), size=[out.shape[-2], out.shape[-1]], mode='nearest')
        shp2 = out2.shape
        out = out1 * out2  # tensor:(2, 16, 64, 64)
        shp3 = out.shape
      
        return out  # DFC

class C3Ghost(C3):  
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneckV2(c_, c_) for _ in range(n)))
