"""
Assignment 2: Quadtree Compression

=== CSC148 Winter 2021 ===
Department of Mathematical and Computational Sciences,
University of Toronto Mississauga

=== Module Description ===
This module contains classes for the file I/O of BMP and QDT
"""

from typing import List, Any

from a2tree import QuadTree

BMP_EXTENSION = '.bmp'
QDT_EXTENSION = '.qdt'


class BaseFile:
    """
    Base class for BMP and QDT files
    """

    filename: str
    offset: int
    width: int
    height: int
    header: List[int]
    body: Any

    def __init__(self, filename: str) -> None:
        """
        Then constructor just initialize the attributes. It doesn't read
        or write the file.
        """
        self.filename = filename
        self.offset = -1
        self.width = -1
        self.height = -1
        self.header = []
        self.body = None

    def read_header(self, int_data: List[int]) -> None:
        """
        Reads the header information from the image and sets the attributes.
        The method's implementation is shared by both BMP and QDT files

        Arguments:
            int_data: a list of ints representing the byte data of an image.

        Important metadata locations:
            Offset: bytes 10-13
            Width: bytes 18-21
            Height: bytes 22-25
        """
        self.offset = int.from_bytes(int_data[10:14], 'little')
        self.width = int.from_bytes(int_data[18:22], 'little')
        self.height = int.from_bytes(int_data[22:26], 'little')
        self.header = int_data[:self.offset]
        return

    def read_body(self, file_data: List[int]) -> None:
        """
        This method reads the body of the file (after header) and assign to
        self.body.
        Implementation in subclasses.
        """
        raise NotImplementedError()

    def load(self) -> None:
        """
        Load the image <self.filename>.
        """
        with open(self.filename, 'rb') as f:
            data = f.read()
        int_data = list(data)
        self.read_header(int_data)
        self.read_body(int_data)
        return

    def set_data(self, offset: int, width: int, height: int,
                 header: List[int], body: Any) -> None:
        """
        Set the values of the attributes.
        """
        self.offset = offset
        self.width = width
        self.height = height
        self.header = header
        self.body = body
        return

    def flatten_body(self) -> List[int]:
        """
        Convert self.body to a linear list of bytes and return the list
        """
        raise NotImplementedError()

    def save(self, filename: str = None) -> None:
        """
        Outputs or 'saves' the image to <filename> if provided.
        If <filename> is not provided, save to self.filename

        Arguments:
            filename: the filename to save the image at
        """
        body_data = self.flatten_body()
        new_image_data = self.header + body_data

        if filename is None:
            filename = self.filename

        print("Saving to file: {}".format(filename))
        with open(f'{filename}', 'wb') as image:
            image.write(bytes(new_image_data))
        return


class BMPFile(BaseFile):
    """
    The BMP file class storing a BMP file

    The body of a BMP file a list of lists (of rgb triples) representing
    the pixels of the image
    """

    def __init__(self, filename: str) -> None:

        super().__init__(filename)

    def read_body(self, file_data: List[int]) -> None:
        """
        Reads all pixels linearly in a list of lists of tuples and sets the body
        attribute.

        Arguments:
            file_data: a list of ints representing the byte data of an image.

        This method reads the pixel data from the image using the
        offset, width, and height attributes of the image.

        At the data offset, each pixel is represented by three consecutive
        integers and there will be (height * width) pixels. Read each pixel
        as a tuple and represent each row of the image as a list of tuples.
        Then represent the image as a list of rows.
        """
        pixel_data = \
            file_data[self.offset: self.offset + self.width * self.height * 3]
        pixel_array = []
        for i in range(self.height):
            pixel_row = []
            for j in range(self.width):
                start_location = (i * self.width + j) * 3
                pixel_info = (pixel_data[start_location],
                              pixel_data[start_location + 1],
                              pixel_data[start_location + 2])
                pixel_row.append(pixel_info)
            pixel_array.append(pixel_row)

        self.body = pixel_array

    def flatten_body(self) -> List[int]:

        output_lst = []
        for row in self.body:
            for pixel in row:
                output_lst.extend(pixel)
        return output_lst


class QuadTreeFile(BaseFile):
    """
    A QDT file that stores a quad tree (same header as the bmp file)

    The body of the QDT file is the QuadTree object serialized into a string
    representing the preorder list of the quadtree.
    """

    def __init__(self, filename: str) -> None:

        super().__init__(filename)

    def read_body(self, int_data: List[int]) -> None:
        """
        Restore the quad tree from the data in the qdt file and assign the
        QuadTree object to self.body
        """
        preorder_list = bytes(int_data[self.offset:]).decode().split(',')
        self.body = QuadTree.restore_from_preorder(preorder_list,
                                                   self.width, self.height)
        print("loaded a tree with size: {}".format(self.body.tree_size()))
        return

    def flatten_body(self) -> List[int]:
        """
        Serializing the quad tree to a list of bytes
        """
        return list(bytes(self.body.preorder(), 'utf-8'))


if __name__ == '__main__':

    import doctest
    doctest.testmod()

    # import python_ta
    # python_ta.check_all()
