"""
Assignment 2: Quadtree Compression

=== CSC148 Winter 2021 ===
Department of Mathematical and Computational Sciences,
University of Toronto Mississauga

=== Module Description ===
This module contains classes implementing the quadtree.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Optional
from copy import deepcopy


# No other imports allowed


def mean_and_count(matrix: List[List[int]]) -> Tuple[float, int]:
    """
    Returns the average of the values in a 2D list
    Also returns the number of values in the list
    """
    total = 0
    count = 0
    for row in matrix:
        for v in row:
            total += v
            count += 1
    return total / count, count


def standard_deviation_and_mean(matrix: List[List[int]]) -> Tuple[float, float]:
    """
    Return the standard deviation and mean of the values in <matrix>

    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Note that the returned average is a float.
    It may need to be rounded to int when used.
    """
    avg, count = mean_and_count(matrix)
    total_square_error = 0
    for row in matrix:
        for v in row:
            total_square_error += ((v - avg) ** 2)
    return math.sqrt(total_square_error / count), avg


class QuadTreeNode:
    """
    Base class for a node in a quad tree
    """

    def __init__(self) -> None:
        pass

    def tree_size(self) -> int:
        raise NotImplementedError

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        raise NotImplementedError

    def preorder(self) -> str:
        raise NotImplementedError


class QuadTreeNodeEmpty(QuadTreeNode):
    """
    An empty node represents an area with no pixels included
    """

    def __init__(self) -> None:
        super().__init__()

    def tree_size(self) -> int:
        """
        Note: An empty node still counts as 1 node in the quad tree
        """
        return 1

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        """
        Convert to a properly formatted empty list
        """
        # Note: Normally, this method should return an empty list or a list of
        # empty lists. However, when the tree is mirrored, this returned list
        # might not be empty and may contain the value 255 in it. This will
        # cause the decompressed image to have unexpected white pixels.
        # You may ignore this caveat for the purpose of this assignment.
        return [[255] * width for _ in range(height)]

    def preorder(self) -> str:
        """
        The letter E represents an empty node
        """
        return 'E'


class QuadTreeNodeLeaf(QuadTreeNode):
    """
    A leaf node in the quad tree could be a single pixel or an area in which
    all pixels have the same colour (indicated by self.value).
    """

    value: int  # the colour value of the node

    def __init__(self, value: int) -> None:
        super().__init__()
        assert isinstance(value, int)
        self.value = value

    def tree_size(self) -> int:
        """
        Return the size of the subtree rooted at this node
        """
        return 1

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        """
        Return the pixels represented by this node as a 2D list

        >>> sample_leaf = QuadTreeNodeLeaf(5)
        >>> sample_leaf.convert_to_pixels(2, 2)
        [[5, 5], [5, 5]]
        >>> sample_leaf.convert_to_pixels(1, 2)
        [[5], [5]]
        """
        if height == 0:
            return [[]]
        return [[self.value] * width for _ in range(height)]

    def preorder(self) -> str:
        """
        A leaf node is represented by an integer value in the preorder string
        """
        return str(self.value)


class QuadTreeNodeInternal(QuadTreeNode):
    """
    An internal node is a non-leaf node, which represents an area that will be
    further divided into quadrants (self.children).

    The four quadrants must be ordered in the following way in self.children:
    bottom-left, bottom-right, top-left, top-right

    (List indices increase from left to right, bottom to top)

    Representation Invariant:
    - len(self.children) == 4
    """
    children: List[Optional[QuadTreeNode]]

    def __init__(self) -> None:
        """
        Order of children: bottom-left, bottom-right, top-left, top-right
        """
        super().__init__()

        # Length of self.children must be always 4.
        self.children = [None, None, None, None]

    def tree_size(self) -> int:
        """
        The size of the subtree rooted at this node.

        This method returns the number of nodes that are in this subtree,
        including the root node.
        """
        num = 1
        for child in self.children:
            num += child.tree_size()
        return num

    def convert_to_pixels(self, width: int, height: int) -> List[List[int]]:
        """
        Return the pixels represented by this node as a 2D list.

        You'll need to recursively get the pixels for the quadrants and
        combine them together.

        Make sure you get the sizes (width/height) of the quadrants correct!
        Read the docstring for split_quadrants() for more info.
        >>> example = QuadTree(0)
        >>> example.root = QuadTreeNodeInternal()
        >>> example.root.children = [QuadTreeNodeEmpty(), QuadTreeNodeLeaf(1), QuadTreeNodeEmpty(),QuadTreeNodeLeaf(2)]
        >>> example.root.convert_to_pixels(3,2)
        [[255, 1, 1], [255, 2, 2]]

        """
        bottom_index = height // 2
        top_index = height - bottom_index

        left_index = width // 2
        right_index = width - left_index

        b_l = self.children[0].convert_to_pixels(left_index, bottom_index)

        b_r = self.children[1].convert_to_pixels(right_index, bottom_index)

        bottom_rows = []
        for i in range(bottom_index):
            row = b_l[i] + b_r[i]
            bottom_rows.append(row)

        t_l = self.children[2].convert_to_pixels(left_index, top_index)

        t_r = self.children[3].convert_to_pixels(right_index, top_index)

        top_rows = []
        for i in range(top_index):
            row = t_l[i] + t_r[i]
            top_rows.append(row)

        if top_rows == [[]] or top_rows == []:
            return bottom_rows

        if bottom_rows == [[]] or bottom_rows == []:
            return top_rows

        pixels = bottom_rows + top_rows
        return pixels

    def preorder(self) -> str:
        """
        Return a string representing the preorder traversal or the tree rooted
        at this node. See the docstring of the preorder() method in the
        QuadTree class for more details.

        An internal node is represented by an empty string in the preorder
        string.
        >>> pixels =[[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
        >>> example = QuadTree(0)
        >>> example.build_quad_tree(pixels)
        >>> example.preorder()
        ',,E,1,E,4,,2,3,5,6,,E,7,E,,E,10,E,13,,8,9,,E,11,E,14,,E,12,E,15'
        """
        if isinstance(self, QuadTreeNodeLeaf) or \
                isinstance(self, QuadTreeNodeEmpty):
            return self.preorder()
        else:
            result = ""
            for child in self.children:
                result += ","
                x = child.preorder()
                result = result + x
        return result

    def restore_from_preorder(self, lst: List[str], start: int) -> int:
        """
        Restore subtree from preorder list <lst>, starting at index <start>
        Return the number of entries used in the list to restore this subtree
        """

        # This assert will help you find errors.
        # Since this is an internal node, the first entry to restore should
        # be an empty string
        assert lst[start] == ''
        index = start + 1
        i = 0
        num = 1
        while i < 4:
            item = lst[index]
            if item == 'E':
                self.children[i] = QuadTreeNodeEmpty()
                num += 1
                index += 1
            elif item == '':
                x = QuadTreeNodeInternal()
                val = x.restore_from_preorder(lst, index)
                index += val
                num += val
                self.children[i] = x
            else:  # item is a leaf
                self.children[i] = QuadTreeNodeLeaf(int(item))
                num += 1
                index += 1
            i += 1
        return num

    def mirror(self) -> None:
        """
        Mirror the bottom half of the image represented by this tree over
        the top half

        Example:
            Original Image
            1 2
            3 4

            Mirrored Image
            3 4 (this row is flipped upside down)
            3 4

        See the assignment handout for a visual example.
        >>> example = QuadTree(0)
        >>> example.build_quad_tree([[1, 2, 3, 4]], True)

        """
        if isinstance(self.children[0], QuadTreeNodeInternal):
            new_top_left = QuadTreeNodeInternal()
            _reverse(self.children[0], new_top_left)
        else:
            if isinstance(self.children[0], QuadTreeNodeLeaf):
                new_top_left = QuadTreeNodeLeaf(self.children[0].value)
            else:  # Empty node
                new_top_left = QuadTreeNodeEmpty()

        if isinstance(self.children[1], QuadTreeNodeInternal):
            new_top_right = QuadTreeNodeInternal()
            _reverse(self.children[1], new_top_right)
        else:
            if isinstance(self.children[1], QuadTreeNodeLeaf):
                new_top_right = QuadTreeNodeLeaf(self.children[1].value)
            else:  # Empty node
                new_top_right = QuadTreeNodeEmpty()

        self.children[2], self.children[3] = new_top_left, new_top_right
        return


class QuadTree:
    """
    The class for the overall quadtree
    """

    loss_level: float
    height: int
    width: int
    root: Optional[QuadTreeNode]  # safe to assume root is an internal node

    def __init__(self, loss_level: int = 0) -> None:
        """
        Precondition: the size of <pixels> is at least 1x1
        """
        self.loss_level = float(loss_level)
        self.height = -1
        self.width = -1
        self.root = None

    def build_quad_tree(self, pixels: List[List[int]],
                        mirror: bool = False) -> None:
        """
        Build a quad tree representing all pixels in <pixels>
        and assign its root to self.root

        <mirror> indicates whether the compressed image should be mirrored.
        See the assignment handout for examples of how mirroring works.
        >>> aa = QuadTree(4)
        >>> aa.build_quad_tree([[1,2,3], [4,5,6], [7,8,9]])
        """
        # print('building_quad_tree...')
        self.height = len(pixels)
        self.width = len(pixels[0])
        self.root = self._build_tree_helper(pixels)
        if mirror:
            self.root.mirror()
        return

    def _build_tree_helper(self, pixels: List[List[int]]) -> QuadTreeNode:
        """
        Build a quad tree representing all pixels in <pixels>
        and return the root

        Note that self.loss_level should affect the building of the tree.
        This method is where the compression happens.

        IMPORTANT: the condition for compressing a quadrant is the standard
        deviation being __LESS THAN OR EQUAL TO__ the loss level. You must
        implement this condition exactly; otherwise, you could fail some
        test cases unexpectedly.

        """
        # Empty Node
        if pixels == [] or pixels == [[]]:
            return QuadTreeNodeEmpty()
        # Leaf Node
        elif len(pixels) == 1 and isinstance(pixels[0], list) and \
                len(pixels[0]) == 1 and isinstance(pixels[0][0], int):
            return QuadTreeNodeLeaf(pixels[0][0])
        # Root Node
        else:
            all_pixels = True
            result = self._split_quadrants(pixels)
            temp = []
            values = []
            count = 0
            for child in result:
                em = False
                if len(child) != 0:
                    empty = [[] * 1 for _ in range(len(child))]
                    if child == empty:
                        em = True

                if em:
                    x = QuadTreeNodeEmpty()
                else:
                    x = self._build_tree_helper(child)

                if isinstance(x, QuadTreeNodeInternal):
                    all_pixels = False
                if isinstance(x, QuadTreeNodeLeaf):
                    values.append(x.value)
                if isinstance(x, QuadTreeNodeEmpty):
                    count += 1

                temp.append(x)

            root = QuadTreeNodeInternal()
            if all_pixels:
                if count == 4:
                    return QuadTreeNodeEmpty()
                else:
                    s_d, mean = standard_deviation_and_mean([values])
                    if s_d <= self.loss_level:
                        return QuadTreeNodeLeaf(round(mean))

            for index in range(4):
                root.children[index] = temp[index]
            return root

    @staticmethod
    def _split_quadrants(pixels: List[List[int]]) -> List[List[List[int]]]:
        """
        Precondition: size of <pixels> is at least 1x1
        Returns a list of four lists of lists, correspoding to the quadrants in
        the following order: bottom-left, bottom-right, top-left, top-right

        IMPORTANT: when dividing an odd number of entries, the smaller half
        must be the left half or the bottom half, i.e., the half with lower
        indices.

        Postcondition: the size of the returned list must be 4
        >>> example = QuadTree(0)
        >>> example._split_quadrants([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        [[[1]], [[2, 3]], [[4], [7]], [[5, 6], [8, 9]]]
        >>> example._split_quadrants([[1,2,3,4],[5,6, 7,8],[9,10,11,12], [13,14,15,16]])
        [[[1, 2], [5, 6]], [[3, 4], [7, 8]], [[9, 10], [13, 14]], [[11, 12], [15, 16]]]
        """
        height = len(pixels)
        width = len(pixels[0])

        horizontal_index = height // 2
        vertical_index = width // 2

        bottom = pixels[:horizontal_index]
        top = pixels[horizontal_index:]

        bottom_left = []
        bottom_right = []
        for i in range(len(bottom)):
            bottom_left.append(bottom[i][:vertical_index])
            bottom_right.append(bottom[i][vertical_index:])

        top_right = []
        top_left = []
        for i in range(len(top)):
            top_right.append(top[i][vertical_index:])
            top_left.append(top[i][:vertical_index])

        result = [bottom_left, bottom_right, top_left, top_right]
        return result

    def tree_size(self) -> int:
        """
        Return the number of nodes in the tree, including all Empty, Leaf, and
        Internal nodes.
        >>> pixels =[[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
        >>> example = QuadTree(0)
        >>> example.build_quad_tree(pixels)
        >>> example.tree_size()
        33
        >>> tree = QuadTree()
        >>> tree.build_quad_tree([[1, 2, 3]])
        >>> tree.tree_size()
        9
        """
        return self.root.tree_size()

    def convert_to_pixels(self) -> List[List[int]]:
        """
        Return the pixels represented by this tree as a 2D matrix
        """
        return self.root.convert_to_pixels(self.width, self.height)

    def preorder(self) -> str:
        """
        return a string representing the preorder traversal of the quadtree.
        The string is a series of entries separated by comma (,).
        Each entry could be one of the following:
        - empty string '': represents a QuadTreeNodeInternal
        - string of an integer value such as '5': represents a QuadTreeNodeLeaf
        - string 'E': represents a QuadTreeNodeEmpty

        For example, consider the following tree with a root and its 4 children
                __      Root       __
              /      |       |        \
            Empty  Leaf(5), Leaf(8), Empty

        preorder() of this tree should return exactly this string: ",E,5,8,E"

        (Note the empty-string entry before the first comma)
        >>> pixels =[[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
        >>> example = QuadTree(0)
        >>> example.build_quad_tree(pixels)
        >>> example.preorder()
        ',,E,1,E,4,,2,3,5,6,,E,7,E,,E,10,E,13,,8,9,,E,11,E,14,,E,12,E,15'
        """
        return self.root.preorder()

    @staticmethod
    def restore_from_preorder(lst: List[str],
                              width: int, height: int) -> QuadTree:
        """
        Restore the quad tree from the preorder list <lst>
        The preorder list <lst> is the preorder string split by comma

        Precondition: the root of the tree must be an internal node (non-leaf)
        """
        tree = QuadTree()
        tree.width = width
        tree.height = height
        tree.root = QuadTreeNodeInternal()
        tree.root.restore_from_preorder(lst, 0)
        return tree


def maximum_loss(original: QuadTreeNode, compressed: QuadTreeNode) -> float:
    """
    Given an uncompressed image as a quad tree and the compressed version,
    return the maximum loss across all compressed quadrants.

    Precondition: original.tree_size() >= compressed.tree_size()

    Note: original, compressed are the root nodes (QuadTreeNode) of the
    trees, *not* QuadTree objects

    >>> pixels = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> orig, comp = QuadTree(0), QuadTree(2)
    >>> orig.build_quad_tree(pixels)
    >>> comp.build_quad_tree(pixels)
    >>> maximum_loss(orig.root, comp.root)
    1.5811388300841898
    >>> p3 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    >>> x = QuadTree(0)
    >>> x.build_quad_tree(p3)
    >>> y = QuadTree(4)
    >>> y.build_quad_tree(p3)
    >>> print(maximum_loss(x.root, y.root))
    2.0615528128088303
    """
    max_loss = 0.0
    if isinstance(original, QuadTreeNodeInternal) \
            and isinstance(compressed, QuadTreeNodeInternal):
        for i in range(4):
            temp = maximum_loss(original.children[i], compressed.children[i])
            if max_loss < temp:
                max_loss = temp
        return max_loss
    elif isinstance(original, QuadTreeNodeLeaf) \
            and isinstance(compressed, QuadTreeNodeLeaf):
        return 0.0
    elif isinstance(original, QuadTreeNodeEmpty) \
            and isinstance(compressed, QuadTreeNodeEmpty):
        return 0.0
    else:  # original is an internal node and compressed is a leaf
        max_loss, mean = _check_deviation(original)
        return max_loss


#  Helper functions

def _reverse(original, new):
    if isinstance(original, QuadTreeNodeEmpty) or \
            isinstance(original, QuadTreeNodeLeaf):
        return
    else:  # original is an Internal node
        n = [2, 3, 0, 1]
        for i in range(4):
            if isinstance(original.children[i], QuadTreeNodeInternal):
                new.children[n[i]] = QuadTreeNodeInternal()
                _reverse(original.children[i], new.children[n[i]])
            elif isinstance(original.children[i], QuadTreeNodeLeaf):
                new.children[n[i]] = QuadTreeNodeLeaf(
                    original.children[i].value)
            else:
                new.children[n[i]] = QuadTreeNodeEmpty()
        return


def _check_deviation(node: QuadTreeNode) -> Tuple[float, float]:
    values = []
    max_loss = 0.0
    for child in node.children:
        if isinstance(child, QuadTreeNodeLeaf):
            values.append(child.value)
        if isinstance(child, QuadTreeNodeInternal):
            max_loss, mean = _check_deviation(child)
            values.append(mean)

    s_d, mean = standard_deviation_and_mean([values])
    if s_d >= max_loss:
        max_loss = s_d
    return max_loss, round(mean)


if __name__ == '__main__':
    import doctest

    doctest.testmod()

    # import python_ta
    # python_ta.check_all()
