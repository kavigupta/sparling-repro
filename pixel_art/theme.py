from matplotlib import colors as mcolors

blue = "#80cdff"
orange = "#ffca80"
green = "#60e37a"
pink = "#ff80b1"
purple = "#bd80ff"

color_series = [blue, orange, green, pink, purple]

color_fpe = orange
color_fne = green
color_ce = pink
color_e2e = blue

color_by_column = {
    "fpe": color_fpe,
    "fne": color_fne,
    "ce": color_ce,
    "e2e_edit": color_e2e,
}


def modify_color(color, saturation_change, value_change):
    m = mcolors.ColorConverter().to_rgb
    rgb = m(color)
    hsv = mcolors.rgb_to_hsv(rgb)
    hsv[1] = 1 - (1 - hsv[1]) * saturation_change
    hsv[2] *= value_change
    color = mcolors.hsv_to_rgb(hsv)
    return color


def darken(color):
    return modify_color(color, 0.5, 0.9)
