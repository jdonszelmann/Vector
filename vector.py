from __future__ import division

from functools import reduce
from typing import List, Iterable
from numbers import Number
from copy import copy, deepcopy
import math
from math import pi


class Vector(object):
    """
    most generic Vector class. can have any dimension

    this vector class is immutable
    """

    def __init__(self, *args: List[int]) -> "Vector":
        self.values = args

    @classmethod
    def checktype(cls, obj: object) -> bool:
        """
        returns True if a class is a vector
        """

        return isinstance(obj, cls)

    @classmethod
    def asVector3(cls, obj: "Vector") -> "Vector":
        """
        transforms a generic vector object into vcetor3
        """
        if not cls.checktype(obj):
            raise TypeError("cant transform non-vector into vector")
        elif len(obj) <= 3:
            return Vector3(*copy(obj.values))
        else:
            raise TypeError(
                "cant generate vector3 object from {}d vector".format(len(obj)))

    @classmethod
    def asVector2(cls, obj: "Vector") -> "Vector":
        """
        transforms a generic vector object into vcetor2
        """
        if not cls.checktype(obj):
            raise TypeError("cant transform non-vector into vector")
        elif len(obj) <= 2:
            return Vector2(*copy(obj.values))
        else:
            raise TypeError(
                "cant generate vector2 object from {}d vector".format(len(obj)))

    def Vector3(self) -> "Vector":
        """
        transforms self into vcetor3
        """

        if len(self) <= 3:
            return Vector3(*copy(self.values))
        else:
            raise TypeError(
                "cant generate vector3 object from {}d vector".format(len(self)))

    def Vector2(self) -> "Vector":
        """
        transforms self into vcetor3
        """

        if len(self) <= 2:
            return Vector2(*copy(self.values))
        else:
            raise TypeError(
                "cant generate vector2 object from {}d vector".format(len(self)))

    def __iter__(self) -> Iterable[float]:
        """
        iterates over the values of the vector.


        in case of a 3d vector: [x,y,z]
        """

        yield from self.values

    def __getitem__(self, sl: slice) -> float:
        """
        returns the n'th dimension of the vector.
        supports slicing

        Vector(1,2,3)[0] -> 1
        """

        return self.values[sl]

    def __setitem__(self, sl: slice, value: float) -> "Vector":
        """
        sets the n'th dimension of the vector.
        supports slicing

        Vector(1,2,3)[0] = 3 -> Vector(3,2,3)
        """
        newvalues = copy(self.values)
        newvalues[sl] = value
        return self.__class__(*newvalues)

    def __repr__(self) -> str:
        """
        generates a string representation of a vector
        """

        return "Vector{}: {}".format(len(self.values), str(self.values))

    def __str__(self) -> str:
        """
        generates a string representation of a vector
        """

        return self.__repr__()

    def print(self) -> "Vector":
        """
        prints this vector and returns self for chaining
        """
        print(self)
        return self

    def add(self, other: (float, "Vector")) -> "Vector":
        """
        adds a scalar value or vector to this vector

        will automatically detect scalar addition or vector addition

        returns self for chaining

        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i + j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i + other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __iadd__(self, other) -> "Vector":
        """
        adds a scalar value or vector to this vector (inplace)

        will automatically detect scalar addition or vector addition
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i + j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i + other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __add__(self, other) -> "Vector":
        """
        adds a scalar value or vector to this vector

        will automatically detect scalar addition or vector addition
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i + j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i + other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def sub(self, other: (float, "Vector")) -> "Vector":
        """
        subtracts a scalar value or vector from this vector

        will automatically detect scalar subtraction or vector subtraction

        returns self for chaining

        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i - j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i - other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __isub__(self, other) -> "Vector":
        """
        subtracts a scalar value or vector from this vector

        will automatically detect scalar subtraction or vector subtraction
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i - j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i - other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __sub__(self, other) -> "Vector":
        """
        subtracts a scalar value or vector from this vector

        will automatically detect scalar subtraction or vector subtraction
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i - j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i - other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def div(self, other: (float, "Vector")) -> "Vector":
        """
        divides this vector by a scalar value or vector

        will automatically detect scalar division or vector division

        returns self for chaining

        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i / j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i / other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __idiv__(self, other) -> "Vector":
        """
        divides this vector by a scalar value or vector (inplace)

        will automatically detect scalar division or vector division
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i / j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i / other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __div__(self, other) -> "Vector":
        """
        divides this vector by a scalar value or vector

        will automatically detect scalar division or vector division
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i / j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i / other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __truediv__(self, other) -> "Vector":
        """
        divides this vector by a scalar value or vector

        will automatically detect scalar division or vector division
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i / j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i / other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __itruediv__(self, other) -> "Vector":
        """
        divides this vector by a scalar value or vector

        will automatically detect scalar division or vector division
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i / j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i / other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def floordiv(self, other: (float, "Vector")) -> "Vector":
        """
        divides this vector by a scalar value or vector (integer division)

        will automatically detect scalar division or vector division

        returns self for chaining

        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i // j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i // other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __ifloordiv__(self, other) -> "Vector":
        """
        divides this vector by a scalar value or vector (inplace) (integer division)

        will automatically detect scalar division or vector division
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i // j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i // other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __floordiv__(self, other) -> "Vector":
        """
        divides this vector by a scalar value or vector (integer division)

        will automatically detect scalar division or vector division
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i // j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i // other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def mod(self, other: (float, "Vector")) -> "Vector":
        """
        performs modulo of this vector by a scalar value or vector

        will automatically detect scalar modulo or vector modulo

        returns self for chaining

        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i % j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i % other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __imod__(self, other) -> "Vector":
        """
        performs modulo of this vector by a scalar value or vector (inplace)

        will automatically detect scalar modulo or vector modulo
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i % j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i % other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __mod__(self, other) -> "Vector":
        """
        performs modulo of this vector by a scalar value or vector

        will automatically detect scalar modulo or vector modulo
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i % j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i % other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def mul(self, other: (float, "Vector")) -> "Vector":
        """
        multiplies this vector by a scalar value or vector

        will automatically detect scalar multiplication or vector multiplication

        returns self for chaining

        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i * j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i * other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __imul__(self, other) -> "Vector":
        """
        multiplies this vector by a scalar value or vector (inplace)

        will automatically detect scalar multiplication or vector multiplication
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i * j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i * other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __mul__(self, other) -> "Vector":
        """
        multiplies this vector by a scalar value or vector

        will automatically detect scalar multiplication or vector multiplication
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i * j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i * other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def pow(self, other: (float, "Vector")) -> "Vector":
        """
        multipexponentiateslies this vector by a scalar value or vector

        will automatically detect scalar exponentiation or vector exponentiation

        returns self for chaining

        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i ** j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i ** other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __ipow__(self, other) -> "Vector":
        """
        exponentiates this vector by a scalar value or vector (inplace)

        will automatically detect scalar exponentiation or vector exponentiation
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i ** j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i ** other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __pow__(self, other) -> "Vector":
        """
        exponentiates this vector by a scalar value or vector

        will automatically detect scalar exponentiation or vector exponentiation
        """

        if self.checktype(other) and len(self) == len(other):
            return self.__class__(
                *map(lambda i, j: i ** j, self, other)
            )
        elif type(other) == Number:
            return self.__class__(
                *map(lambda i: i ** other, self)
            )
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __len__(self) -> int:
        """
        returns the number of dimensions of the vector
        """
        return len(self.values)

    def length(self) -> int:
        """
        returns the length of the vector
        """
        return sum(map(lambda i: i ** 2, self))**0.5

    def __abs__(self) -> int:
        """
        alias for Vector.length
        """
        return self.length()

    def distance(self, other: "Vector") -> float:
        """
        calculates the distance between this vector an another vector
        """

        if self.checktype(other) and len(self) == len(other):
            return sum(map(lambda i, j: (i - j) ** 2, self, other))**0.5
        else:
            raise TypeError(
                "can't perform operation on vector with a larger dimension with a vector with a smaller dimension")

    def __matmul__(self, other: "Vector") -> float:
        """
        alias for Vector.distance
        """
        return self.distance(other)

    def dotproduct(self, other: "Vector") -> float:
        """
        calculates the dot product of this vector and another
        """
        return sum(map(lambda i, j: i * j, self, other))

    def magnitude(self) -> float:
        """
        alias for Vector.length
        """
        return self.length()

    def setmagnitude(self, magnitude: float) -> "Vector":
        """
        sets the magnitude of a vector
        """
        return self.__class__(
            *map(lambda i: i/self.magnitude() * magnitude, self)
        )

    def limit(self, limit: float):
        """
        limits the length of the vector. if it's larger reduce magnitude
        """

        if self.magnitude() > limit:
            return self.setmagnitude(limit)
        return self.copy()

    def normalize(self) -> "Vector":
        """
        sets the magnitude of this vector to 1
        """
        return self.setmagnitude(1)

    def copy(self) -> "Vector":
        """
        copies this vector
        """
        return self.__class__(copy(self.values))

    def deepcopy(self) -> "Vector":
        """
        copies this vector
        """
        return self.__class__(deepcopy(self.values))

    def __copy__(self) -> "Vector":
        """
        copies this vector
        """
        return self.__class__(copy(self.values))

    def __deepcopy__(self) -> "Vector":
        """
        copies this vector
        """
        return self.__class__(deepcopy(self.values))

    def __hash__(self):
        return reduce(lambda acc, val: acc ^ val, self)

    def angle(self, other):
        """
        calculates the angle between 2 vectors
        """

        return math.acos(self.inproduct(other) / (self.magnitude * other.magnitude))

    def __round__(self, n: int) -> "Vector":
        """
        rounds a vector
        """
        return self.__class__(*map(lambda i: round(i, n), self))

    def round(self, n: int) -> "Vector":
        """
        rounds a vector
        """
        return self.__class__(*map(lambda i: round(i, n), self))

    def __eq__(self, other: "Vector")-> bool:
        return all(map(lambda i, j: i == j, self, other))


class Vector2(Vector):
    def __init__(self, x: float = 0, y: float = 0):
        super().__init__(x, y)

    @property
    def x(self):
        """
        return the 1st dimension of this vector.
        """
        return self[0]

    @property
    def y(self):
        """
        return the 2nd dimension of this vector.
        """
        return self[1]

    def __getitem__(self, sl: (slice, str, int)) -> float:
        """
        add x,y functionality to Vector getitem.
        """

        if sl == "x":
            return self.x
        elif sl == "y":
            return self.y
        else:
            return super().__getitem__(sl)

    def isperpendicular(self, other: "Vector") -> bool:
        """
        find if the angle between 2 vectors is 90 degrees
        (should be in main vector class?)
        """

        return self.dotproduct(other) == 0

    def rotate(self, angle: float) -> "Vector":
        """
        rotate a 2d vector around the origin
        """

        def dot(matrix):
            """
            Matrix must be ordered in the following way:
            Each element in main list is a row
            [[0,0], [0,0]]
            """
            new = [0]*len(self)
            for indexrow, row in enumerate(matrix):
                intermediary = 0
                for indexcolumn, elem in enumerate(row):
                    intermediary += elem * self[indexcolumn]
                new[indexrow] = intermediary
            return new

        new = Vector2(*dot(
            [[math.cos(angle), -math.sin(angle)],
             [math.sin(angle), math.cos(angle)]]
        ))

        return new


class Vector3(Vector2):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        super(Vector2, self).__init__(x, y, z)

    @property
    def z(self):
        """
        return the 3rd dimension of this vector.
        """

        return self[2]

    def __getitem__(self, sl: (slice, str, int)) -> float:
        """
        add z functionality to Vector getitem.
        """

        if sl == "z":
            return self.z
        else:
            return super().__getitem__(sl)

    def crossproduct(self, other: "Vector") -> "Vector":
        """
        calculates the cross product of this vector and another
        """
        return self.__class__(
            self.y * other.z - other.y * self.z,
            self.z * other.x - self.x * other.z,
            self.x * other.y - other.x * self.y
        )

    def rotate(self, angle: float, axis='x') -> "Vector":
        """

        rotate a vector a radians around the x,y or z axis. note: different than 2d rotate

        """

        def dot(matrix):
            """
            Matrix must be ordered in the following way:
            Each element in main list is a row
            [[0,0,0], [0,0,0], [0,0,0]]
            """
            new = [0]*len(self)
            for indexrow, row in enumerate(matrix):
                intermediary = 0
                for indexcolumn, elem in enumerate(row):
                    intermediary += elem * self[indexcolumn]
                new[indexrow] = intermediary
            return new

        if axis == 'x':
            X_AXIS = [[1, 0, 0],
                      [0, math.cos(angle), -math.sin(angle)],
                      [0, math.sin(angle), math.cos(angle)]]
            new = Vector3(*dot(X_AXIS))
        elif axis == 'y':
            Y_AXIS = [[math.cos(angle), 0, math.sin(angle)],
                      [0, 1, 0],
                      [-math.sin(angle), 0, math.cos(angle)]]
            new = Vector3(*dot(Y_AXIS))
        elif axis == 'z':
            Z_AXIS = [[math.cos(angle), -math.sin(angle), 0],
                      [math.sin(angle), math.cos(angle), 0],
                      [0, 0, 1]]
            new = Vector3(*dot(Z_AXIS))
        elif axis is str:
            raise ValueError("Axis not supported, only x-, y- and z-axis")
        else:
            raise TypeError("Axis not supported, only x-, y- and z-axis")

        return new
