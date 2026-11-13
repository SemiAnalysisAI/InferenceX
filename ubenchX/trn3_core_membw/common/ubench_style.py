"""Shared visual style for the Ascend-ubench figures; data stays in each generator."""
from pathlib import Path
import os
import matplotlib as mpl
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, to_rgb

HUAWEI = '#C7000B'
NVIDIA = '#76B900'
AMD = '#333333'
INK = '#172F3B'
MUTED = '#5E707A'
GRID = '#E1E8EB'
SURFACE = '#FFFFFF'
RASTER_DPI = 400
HUAWEI_TONES = ('#7F0011', '#C7000B', '#DA4148', '#E8767B', '#F0B0B4', '#F8DBDD')
NVIDIA_TONES = ('#3C6500', '#558A00', '#76B900', '#97CF40', '#BCE18A', '#E0F0C2')
AMD_TONES = ('#333333', '#626262', '#929292')
CHIP_COLORS = {
    'ascend950pr': HUAWEI, '950pr': HUAWEI, 'npu': HUAWEI,
    'ascend950dt': HUAWEI_TONES[0], '950dt': HUAWEI_TONES[0],
    'h100': NVIDIA, 'h200': NVIDIA_TONES[0], 'b200': NVIDIA_TONES[3],
    'gb200': NVIDIA_TONES[3], 'b300': NVIDIA_TONES[1], 'gb300': NVIDIA_TONES[1],
    'mi300x': AMD_TONES[2], 'mi325x': AMD_TONES[1], 'mi355x': AMD_TONES[0],
}


def configure(font_size=11, dpi=180):
    font_dir = Path(os.environ.get('OUTFIT_FONT_DIR', Path.home() / 'Library/Fonts'))
    weights = {'regular': 'Regular', 'medium': 'Medium', 'semibold': 'SemiBold', 'bold': 'Bold'}
    fonts = {}
    for weight, name in weights.items():
        path = font_dir / f'Outfit-{name}.ttf'
        if not path.is_file():
            raise FileNotFoundError(f'Local Outfit font missing: {path}; set OUTFIT_FONT_DIR')
        fm.fontManager.addfont(str(path))
        fonts[weight] = fm.FontProperties(fname=str(path))
    mpl.rcParams.update({
        'font.family': 'Outfit', 'font.size': font_size,
        'text.color': INK, 'axes.labelcolor': MUTED,
        'axes.edgecolor': GRID, 'xtick.color': MUTED, 'ytick.color': MUTED,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'figure.dpi': dpi, 'savefig.dpi': RASTER_DPI,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'grid.color': GRID, 'grid.linewidth': .7,
        'axes.axisbelow': True, 'legend.frameon': False,
        'xtick.major.size': 0, 'ytick.major.size': 0,
        'svg.fonttype': 'path', 'pdf.fonttype': 42,
        'mathtext.fontset': 'custom', 'mathtext.default': 'regular',
        'mathtext.rm': 'Outfit', 'mathtext.it': 'Outfit', 'mathtext.cal': 'Outfit',
        'mathtext.tt': 'Outfit', 'mathtext.sf': 'Outfit',
        'mathtext.bf': 'Outfit:bold', 'mathtext.bfit': 'Outfit:bold',
    })
    return fonts


def family_cmap(vendor='huawei'):
    colors = {'huawei': HUAWEI_TONES, 'nvidia': NVIDIA_TONES, 'amd': AMD_TONES}[vendor]
    return LinearSegmentedColormap.from_list(f'ubench_{vendor}', [SURFACE, *reversed(colors)])


def contrast_text(fill):
    """Choose dark or white text by relative luminance, independently of theme."""
    def luminance(color):
        linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in to_rgb(color)]
        return sum(a * b for a, b in zip(linear, (.2126, .7152, .0722)))
    light = '#FFFFFF'
    dark = '#172F3B'
    y = luminance(fill)
    contrast_light = 1.05 / (y + .05)
    yd = luminance(dark)
    contrast_dark = (max(y, yd) + .05) / (min(y, yd) + .05)
    return light if contrast_light > contrast_dark else dark
