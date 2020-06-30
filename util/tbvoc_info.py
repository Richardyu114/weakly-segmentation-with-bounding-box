"""
list the class and corresponding color info
there are just two main classes in this database:
background: 0
TBbacillus: 1
ignore: 2  the ignore regions in SDI paper

RGB encode reference:
 https://en.m.fontke.com/tool/rgb/800080/

"""
# tbvoc_classes = {
#         'background': 0,
#         'TBbacillus': 1,
#         'ignore': 2}


# colors = {
#     0: [0, 0, 0],
#     1: [128, 0, 0],
#     2: [252, 230, 201]
# }


# colors_map = [
#         [0, 0, 0],
#         [128, 0, 0],
#         [252, 230, 201]
# ]

# don't use original ignore area driectly
# just shrink the area intentionally
tbvoc_classes = {
        'background': 0,
        'TBbacillus': 1
}


colors = {
    0: [0, 0, 0],
    1: [128， 0， 0]
}


colors_map = [
        [0, 0, 0],
        [128， 0， 0]
]
