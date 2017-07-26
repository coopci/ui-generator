# -*- coding: utf-8 -*-
import numpy as np
from funcs import reorder_matches, drawOrderedMatches, drawMatches
import cv2
from matplotlib import pyplot as plt

css_variables_tempplate = """:root {
  --main-bg-img: url("{{main-bg-img}}");
  --main-bg-width: {{main-bg-width}}px;
  --main-bg-height: {{main-bg-height}}px;
}

"""
backgroud_template = """.bgimg {
    background-image: var(--main-bg-img);
	height: var(--main-bg-height);
	width: var(--main-bg-width);
}

"""

page_template = """<html>
<style type="text/css">
{{variables}}

.bgimg {
    background-image: var(--main-bg-img);
	height: var(--main-bg-height);
	width: var(--main-bg-width);
}

{{styles}}

</style>
<body style="margin:0px;" >
<div class="bgimg" >
    {{buttons}}
</div>
</body>
</html>
"""

def render_variables(variables):
    ret = css_variables_tempplate
    for k,v in variables.items():
        place_holder = "{{" + k +"}}"
        ret = ret.replace(place_holder, str(v))
    return ret



def get_dimension(kp1, kp2, matches):
    """
    把kp1当query， 把kp2 当train。
    :param kp1:
    :param kp2:
    :param matches:
    :return:    (query_width, query_height) (train_width, train_height)
    """
    query_top = 0
    query_left = 0
    query_bottom = 0
    query_right = 0

    train_top = 0
    train_left = 0
    train_bottom = 0
    train_right = 0

    if len(matches) > 0:
        mat = matches[0]
        (x1, y1) = kp1[mat.queryIdx].pt
        (x2, y2) = kp2[mat.trainIdx].pt

        query_left = query_right = x1
        query_top = query_bottom = y1

        train_left = train_right = x2
        train_top = train_bottom = y2


    for mat in matches:

        # x - columns
        # y - rows
        (x1, y1) = kp1[mat.queryIdx].pt
        (x2, y2) = kp2[mat.trainIdx].pt

        if x1 > query_right:
            query_right = x1
            train_right = x2

        if x1 < query_left:
            query_left = x1
            train_left = x2

        if y1 > query_bottom:
            query_bottom = y1
            train_bottom = y2
        if y1 < query_top:
            query_top = y1
            train_top = y2


        if x2 > train_right:
            train_right = x2

        if x2 < train_left:
            train_left = x2

        if y2 > train_bottom:
            train_bottom = y2

        if y2 < train_top:
            train_top = y2







    return (query_right- query_left, query_bottom - query_top), (train_right - train_left, train_bottom - train_top)

def genHTML(img1, kp1, filepath1, img2, kp2, filepath2, ordered_matches, outfilepath):
    """
    把 filepath2 作为整个背景。

    :param img1:
    :param kp1:
    :param filepath1:
    :param img2:
    :param kp2:
    :param filepath2:
    :param ordered_matches:
    :param outfilepath:
    :return:
    """
    htmlfile = open(outfilepath, "w")


    variables = {}
    variables["main-bg-img"] = filepath2
    variables["main-bg-height"] = img2.shape[0]
    variables["main-bg-width"] = img2.shape[1]

    rendered_variables = render_variables(variables)

    styles = ""


    asset_width = img1.shape[1]
    asset_height = img1.shape[0]

    occurs_in_layout = []
    for c in ordered_matches.keys():
        # 每一个c 代表一个被找到的元素，
        # 这里要确定每个元素的位置和大小。

        matches = ordered_matches[c]
        # 找到 matches中的train和query 的宽和高
        # train 的宽  / query 的宽   * asset_width  就是这个元素的宽
        # train 的高 / query 的高   * asset_height  就是这个元素的高

        dim_asset, dim_layout = get_dimension(kp1, kp2, matches)

        width_in_layout = dim_layout[0] / dim_asset[0] * asset_width
        height_in_layout = dim_layout[1] / dim_asset[1] * asset_height


        #width_in_layout = asset_width
        #height_in_layout = asset_height


        print width_in_layout, height_in_layout

        top_in_layout = 0
        left_in_layout = 0
        for mat in matches:
            (x1, y1) = kp1[mat.queryIdx].pt
            (x2, y2) = kp2[mat.trainIdx].pt

            # top_in_layout = y2 - y1 * dim_layout[1] / dim_asset[1]
            # left_in_layout = x2 - x1 * dim_layout[0] / dim_asset[0]

            top_in_layout += (y2 - y1* dim_layout[1] / dim_asset[1])
            left_in_layout += (x2 - x1* dim_layout[0] / dim_asset[0])

            #top_in_layout += (y2 - y1 )
            #left_in_layout += (x2 - x1)

        top_in_layout /= len(matches)
        left_in_layout /= len(matches)




        print top_in_layout, left_in_layout

        occurs_in_layout.append({"top": int(top_in_layout),
                                 "left": int(left_in_layout),
                                 "width": int(width_in_layout),
                                 "height": int(height_in_layout),
                                 })


    rendered_buttons = ""
    for b in occurs_in_layout:
        template = """<button type = "button" class ="myButton" id="myButton1" style="padding:0px; background: url({{imgurl}}); border: none; position:absolute; top: {{top}}px; left: {{left}}px; width:{{width}}px; height: {{height}}px" > </button> """
        btn = template.replace("{{imgurl}}", filepath1)
        btn = btn.replace("{{top}}", str(b["top"]))
        btn = btn.replace("{{left}}", str(b["left"]))
        btn = btn.replace("{{width}}", str(b["width"]))
        btn = btn.replace("{{height}}", str(b["height"]))
        rendered_buttons += btn + "\n"



    page = page_template
    page = page.replace("{{variables}}", rendered_variables)

    page = page.replace("{{styles}}", styles)
    page = page.replace("{{buttons}}", rendered_buttons)

    htmlfile.write(page)

    htmlfile.close()
    pass



