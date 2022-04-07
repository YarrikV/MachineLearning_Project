triangles = ['A13', 'A14', 'A15', 'A1AB', 'A1CD', 'A23', 'A23_yellow', 
  'A25', 'A29', 'A31', 'A51', 'A7A', 'A7B', 'B15A', 'B17']

red_circles = ['B19', 'C11', 'C21', 'C23', 'C29', 'C3', 'C31', 'C35', 'C43','F4a']

diamonds = ['B9', 'B11']
forbidden = ['C1']
stop = ['B5']
other = ['C37', 'F1','F13',  'F1a_h', 'F21', 'F23A', 'F25', 'F27', 'F29', 'F31', 'F33_34', 'F35', 'F3a_h', 'F4b', 'F41', 'F43','Handic',  'begin', 'e0c', 'end', 'lang', 'm']
rectangles_down = ['F12a', 'F12b']
rectangles_up = ['B21', 'E9a', 'E9a_miva', 'E9b', 'E9cd', 'E9e','F45', 'F47','F59', 'X']
squares = ['F19','F49','F50','F87']

blue_circles = ['D10', 'D1a', 'D1b', 'D1e', 'D5', 'D7', 'D9']
red_blue_circles = ['E1', 'E3', 'E5', 'E7']
reversed_triangles = ['B1', 'B3',  'B7']

SUPERCLASSES = ["blue_circles", "diamonds", "forbidden", "other", "rectangles_down", "rectangles_up", "red_blue_circles", "red_circles", "reversed_triangles", "squares", "stop", "triangles"]
CLASSESFORSUPERCLASS = [blue_circles, diamonds, forbidden, other, rectangles_down, rectangles_up, red_blue_circles, red_circles, reversed_triangles, squares, stop, triangles]

CLASSES = ['A13', 'A14', 'A15', 'A1AB', 'A1CD', 'A23',
 'A23_yellow', 'A25', 'A29', 'A31', 'A51', 'A7A', 'A7B',
 'B1', 'B11', 'B15A', 'B17', 'B19', 'B21', 'B3', 'B5', 'B7', 
 'B9', 'C1', 'C11', 'C21', 'C23', 'C29', 'C3', 'C31', 'C35', 
 'C37', 'C43', 'D10', 'D1a', 'D1b', 'D1e', 'D5', 'D7', 'D9', 
 'E1', 'E3', 'E5', 'E7', 'E9a', 'E9a_miva', 'E9b', 'E9cd', 'E9e', 
 'F1', 'F12a', 'F12b', 'F13', 'F19', 'F1a_h', 'F21', 'F23A', 'F25', 
 'F27', 'F29', 'F31', 'F33_34', 'F35', 'F3a_h', 'F41', 'F43', 'F45', 
 'F47', 'F49', 'F4a', 'F4b', 'F50', 'F59', 'F87', 'Handic', 'X', 
 'begin', 'e0c', 'end', 'lang', 'm']

# SUPERCLASSES_ = []
# for c in CLASSES:
#     for i, sc in enumerate(CLASSESFORSUPERCLASS):
#         if c in sc:
#             SUPERCLASSES_.append(i)

# print(SUPERCLASSES_)

# Classes as encoded have a respective superclass as encoded
SUPERCLASSES_v2 = [11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 8, 1, 11, 
    11, 7, 5, 8, 10, 8, 1, 2, 7, 7, 7, 7, 7, 7, 7, 3, 7, 0, 0, 0, 0, 0, 
    0, 0, 6, 6, 6, 6, 5, 5, 5, 5, 5, 3, 4, 4, 3, 9, 3, 3, 3, 3, 3, 3, 3, 
    3, 3, 3, 3, 3, 5, 5, 9, 7, 3, 9, 5, 9, 3, 5, 3, 3, 3, 3, 3]
    


