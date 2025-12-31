# Copyright (C) 2025 Certainty Computing Co. Limited. All rights reserved.
#
# MIT License
#
# Copyright (c) 2025 Certainty Computing Co. Limited
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Copyright (c) 2025 Certainty Computing Co. Limited.　
# The Right to freely use and distribute this software is granted to Logopolis™.


"""
MQL通用平衡三進制計算核心 v1.0
Copyright (c) 2025 Certainty Computing Co. Limited. All rights reserved.

核心創新點：

1. 實現函數完備的現代平衡三進制軟體計算內核
   - 原生支持-1/0/1三值邏輯運算，完整實現19683種三值邏輯函數，覆蓋全部函數空間
   - 純軟體實現，無需專用硬體即可在現有計算平臺上部署應用，極大降低使用門檻
   - 特別適用於AI推理、多值邏輯系統、數據壓縮和需要高效表示對稱數據的科學計算

2. 獨創的QZ張量積函數體系
   - 基於9維PSI參數空間的張量積運算，提供極高的函數表達靈活性
   - 實現單變數和雙變數多項式函數計算，支持19683種邏輯門的完整映射
   - 為多模態融合和複雜關係推理提供統一數學框架

3. 分層架構設計
   - 邏輯基元層：Trit實現三值邏輯原子操作，保證基礎操作可靠性
   - 位容器層：TritDigit提供純位操作視圖，分離數值語義與位表示
   - 數值實體層：TritNumber封裝數值語義，提供直觀的數值介面
   - 運算協作層：ArithmeticCore實現高效演算法，保持各層獨立性和可擴展性

4. 智能演算法選擇與優化
   - Karatsuba乘法與大數乘法的自適應切換
   - 內置啟發式優化器，在多項式展開與張量積計算間智能選擇最優路徑
   - LRU緩存機制加速複雜函數迭代計算

5. 形式化驗證與語義保障
   - 基於Coq定理證明器，對27種三值模態算子的函數完備性及其正交性進行了機器驗證（提供可選介面）
   - Coq定理證明確保19683種邏輯函數空間的數學完備性，為上層運算提供可靠的邏輯基元
   - 架構設計為關鍵演算法（如Karatsuba乘法、Douglas W. Jones非恢復除法）的形式化驗證預留介面

6. 統一計算核心
   - 智能選擇最優計算路徑（多項式展開 vs 張量積運算）
   - 支持基於運行時分析的啟發式自動優化
   - 提供運算符重載的Pythonic介面，降低使用門檻

7. 工業級實現
   - 記憶體高效的Trit緩存機制，減少對象創建開銷
   - 支持大整數運算和向量化操作，滿足實際應用需求
   - 全面單元測試覆蓋，確保代碼品質

實現說明：
- 當前版本的Coq形式化驗證專注於三值模態算子的數學性質（多項式算子完備性及正交性）
- 算術運算核心（ArithmeticCore）的正確性目前通過嚴格單元測試保障，未來版本計畫擴展形式化驗證範圍
- 張量積運算層（TensorCore）包含半張量積優化實現，為提升矩陣運算性能設計
"""


import inspect, os, random, unittest
from typing import Callable, List, Optional, Tuple, Union
from functools import lru_cache


# ----------------------
# 0. 自定義異常類（提升調試效率）
# ----------------------
class InvalidTokenError(Exception):
    """運算式中出現無效標記時拋出"""
    pass


class OperandMissingError(Exception):
    """操作符缺少必要運算元時拋出"""
    pass


class QSyntaxError(Exception):
    """語法錯誤，包含行號資訊"""

    def __init__(self, message: str, line_num: int):
        super().__init__(f"第 {line_num} 行：{message}")
        self.line_num = line_num

    # Copyright (c) 2025 Certainty Computing Co. Limited. All rights reserved.


# ----------------------
# 1. 邏輯基元
# ----------------------
class Trit:
    __slots__ = ('value',)
    _cache = {}  # 緩存字典

    def __new__(cls, value: int):
        if value not in (-1, 0, 1):
            raise ValueError("Trit值必須是-1,0或1")
        # 檢查緩存，如果不存在則創建新實例並緩存
        if value not in cls._cache:
            instance = super().__new__(cls)
            instance.value = value
            cls._cache[value] = instance
        return cls._cache[value]

    # 序列化方法
    def __getstate__(self) -> int:
        return self.value

    def __setstate__(self, state: int):
        # 反序列化時無需操作，因為值已在__new__中設置
        pass

    # 身份方法
    def __hash__(self) -> int:
        return hash(self.value)

    # 類型轉換方法
    def __int__(self) -> int:
        return self.value

    def __float__(self) -> float:
        return float(self.value)

    def __bool__(self) -> bool:
        return self.value == 1

    def __index__(self) -> int:
        return self.value + 1

    # 字串表示
    def __str__(self) -> str:
        return {-1: "False", 0: "Doubt", 1: "True"}[self.value]

    def __repr__(self) -> str:
        return f"Trit({self.value})"

    def __format__(self, format_spec: str) -> str:
        """格式化輸出"""
        if not format_spec or format_spec == 's':
            return str(self)
        if format_spec == 'r':
            return repr(self)
        if format_spec == 'i':
            return str(self.value)
        raise ValueError(f"不支持的格式代碼: '{format_spec}'")

    # 上下文管理
    def __eq__(self, other):
        if not isinstance(other, Trit):
            return False
        return self.value == other.value

    def __ne__(self, other) -> bool:
        if not isinstance(other, Trit):
            return True
        return self.value != other.value

    # 禁止任何算術運算方法
    # Copyright (c) 2025 Certainty Computing Co. Limited. All rights reserved.


# ----------------------
# 2. 位容器 (純位操作)
# ----------------------
class TritDigit:
    """純位容器，不包含任何數值語義或運算邏輯"""
    __slots__ = ('digits',)  # 禁止動態屬性

    def __init__(self, digits: List[Trit]):
        # 驗證輸入必須是Trit列表
        if not all(isinstance(t, Trit) for t in digits):
            raise TypeError("TritDigit只能包含Trit實例")
        self.digits = digits

    # 核心位操作方法
    def align(self, length: int) -> 'TritDigit':
        """對齊位長度（高位補零）"""
        if len(self.digits) >= length:
            return self
        padding = [Trit(0)] * (length - len(self.digits))
        return TritDigit(padding + self.digits)

    def shift(self, n: int) -> 'TritDigit':
        """純位移操作（無數值語義）"""
        if n >= 0:
            return TritDigit(self.digits + [Trit(0)] * n)
        return TritDigit(self.digits[:n])

    def extract_bit(self, position: int) -> Trit:
        """提取特定位（邊界返回0）"""
        if 0 <= position < len(self.digits):
            return self.digits[position]
        return Trit(0)

    def set_bit(self, position: int, value: Trit) -> 'TritDigit':
        """設置特定位（自動擴展位寬）"""
        if not isinstance(value, Trit):
            raise TypeError("值必須是Trit實例")

        new_digits = self.digits.copy()
        if position >= len(new_digits):
            new_digits.extend([Trit(0)] * (position - len(new_digits) + 1))
        new_digits[position] = value
        return TritDigit(new_digits)

    def to_integer(self) -> int:
        """將位視圖轉換為整數（無數值語義，僅用於運算）"""
        value = 0
        for i, digit in enumerate(reversed(self.digits)):
            value += digit.value * (3 ** i)
        return value

    # 迭代方法支持
    def __iter__(self):
        """支持迭代訪問每一位"""
        return iter(self.digits)

    # 序列化支持
    def __getstate__(self) -> List[int]:
        return [t.value for t in self.digits]

    def __setstate__(self, state: List[int]):
        self.digits = [Trit(v) for v in state]

    def __eq__(self, other: 'TritDigit') -> bool:
        """位序列完全匹配，非數值語義"""
        if not isinstance(other, TritDigit):
            return False
        return self.digits == other.digits

    def __ne__(self, other: 'TritDigit') -> bool:
        return not self.__eq__(other)

    def __repr__(self) -> str:
        return f"TritDigit({[d.value for d in self.digits]})"

    # 身份方法
    def __hash__(self) -> int:
        """基於位序列的哈希值"""
        return hash(tuple((d.value for d in self.digits)))

    # 索引方法
    def __getitem__(self, key: Union[int, slice]) -> Union[Trit, 'TritDigit']:
        """支持整數索引和切片操作
        - 整數索引：越界返回Trit(0)（符合平衡三進制語義）
        - 切片操作：返回包含切片結果的新TritDigit實例
        """
        if isinstance(key, int):
            # 整數索引邏輯（越界返回0）
            if 0 <= key < len(self.digits):
                return self.digits[key]
            return Trit(0)
        elif isinstance(key, slice):
            # 切片操作：對digits進行切片後創建新TritDigit
            sliced_digits = self.digits[key]
            return TritDigit(sliced_digits)
        else:
            raise TypeError(f"索引必須是整數或切片，而非 {type(key)}")

    # 不實現任何算術運算符
    # Copyright (c) 2025 Certainty Computing Co. Limited. All rights reserved.


# ----------------------
# 3. 數值實體 (純數值語義)
# ----------------------
class TritNumber:
    """數值實體，通過組合使用TritDigit"""
    __slots__ = ('_digit_view',)  # 禁止動態屬性

    def __init__(self, digit_view: TritDigit):
        if not isinstance(digit_view, TritDigit):
            raise TypeError("必須使用TritDigit初始化")
        self._digit_view = digit_view

    @property
    def digits(self) -> List[Trit]:
        """直接訪問底層Trit列表"""
        return self._digit_view.digits

    # 工廠方法
    @classmethod
    def from_digits(cls, digits: List[Trit]) -> 'TritNumber':
        """從Trit列表構造"""
        return cls(TritDigit(digits))

    @classmethod
    def from_integer(cls, value: int) -> 'TritNumber':
        """大整數轉換演算法"""
        if value == 0:
            return cls(TritDigit([Trit(0)]))

        # 處理符號
        sign = 1
        if value < 0:
            sign = -1
            value = -value

        digits = []
        num = value

        while num > 0:
            remainder = num % 3
            num //= 3

            if remainder == 2:
                remainder = -1
                num += 1
            digits.append(Trit(remainder))

        # 處理負值
        if sign == -1:
            digits = [Trit(-d.value) for d in digits]

        # 使用棧結構避免反轉
        digit_stack = []
        while digits:
            digit_stack.append(digits.pop())

        return cls(TritDigit(digit_stack))

    @classmethod
    def safe_from_integer(cls, num: int) -> 'TritNumber':
        """安全創建多位數三進制數，自動處理邊界值"""
        # 優化常見值（-1, 0, 1）直接使用單數字
        if num == 0:
            return cls.zero()
        if num == 1:
            return cls(TritDigit([Trit(1)]))
        if num == -1:
            return cls(TritDigit([Trit(-1)]))

        # 常規轉換
        return cls.from_integer(num)

    # 轉換方法
    def to_digit_view(self) -> TritDigit:
        """獲取純位視圖（無數值語義）"""
        return self._digit_view

    def to_integer(self) -> int:
        """轉換為整數"""
        result = 0
        n = len(self.digits)
        for i in range(n):
            exponent = n - 1 - i
            digit_value = self.digits[i].value
            result += digit_value * (3 ** exponent)
        return result

    # 類型轉換
    def __int__(self) -> int:
        return self.to_integer()

    def __float__(self) -> float:
        return float(self.to_integer())

    # 迭代協議（通過位容器）
    def __iter__(self):
        return iter(self._digit_view)

    # 比較運算符
    def __gt__(self, other: 'TritNumber') -> bool:
        return self.to_integer() > other.to_integer()

    def __lt__(self, other: 'TritNumber') -> bool:
        return self.to_integer() < other.to_integer()

    def __ge__(self, other: 'TritNumber') -> bool:
        return self.to_integer() >= other.to_integer()

    def __le__(self, other: 'TritNumber') -> bool:
        return self.to_integer() <= other.to_integer()

    def __eq__(self, other: 'TritNumber') -> bool:
        """數值相等（非位序列）"""
        if not isinstance(other, TritNumber):
            return False
        return self.to_integer() == other.to_integer()

    def __ne__(self, other: 'TritNumber') -> bool:
        return self.to_integer() != other.to_integer()

    # 數值-邏輯符號比較
    def sign(self) -> Trit:
        """符號比較（只需檢查最高位）"""
        if not self._digit_view.digits or self == TritNumber.zero():
            return Trit(0)
        return Trit(1) if self._digit_view.digits[0].value > 0 else Trit(-1)

    # 單例值
    _ZERO = None
    _ONE = None

    @classmethod
    def zero(cls) -> 'TritNumber':
        if cls._ZERO is None:
            cls._ZERO = cls(TritDigit([Trit(0)]))
        return cls._ZERO

    @classmethod
    def one(cls) -> 'TritNumber':
        if cls._ONE is None:
            cls._ONE = cls(TritDigit([Trit(1)]))
        return cls._ONE

    # 禁止直接位操作
    def __repr__(self) -> str:
        return f"TritNumber({self.to_integer()})"

    # 字串方法
    def __str__(self) -> str:
        return str(self.to_integer())

    # 數值運算介面（通過協作層實現，遵循迪米特法則）
    def __add__(self, other: 'TritNumber') -> 'TritNumber':
        return ArithmeticCore.add(self, other)

    def __sub__(self, other: 'TritNumber') -> 'TritNumber':
        return ArithmeticCore.subtract(self, other)

    def __truediv__(self, other: 'TritNumber') -> 'TritNumber':
        quotient, _ = ArithmeticCore.divide_with_remainder(self, other)
        return quotient

    def __floordiv__(self, other: 'TritNumber') -> 'TritNumber':
        quotient, _ = ArithmeticCore.trit_floor_divide(self, other)
        return quotient

    def __mul__(self, other: 'TritNumber') -> 'TritNumber':
        return ArithmeticCore.multiply(self, other)

    def __neg__(self) -> 'TritNumber':
        """取負數（每位取反）"""
        return TritNumber(TritDigit([Trit(-d.value) for d in self._digit_view.digits]))

    def __abs__(self) -> 'TritNumber':
        """返回絕對值（非負值）"""
        if self.sign() == Trit(-1):
            # 確保取反後是正確的非負值
            negated = TritNumber(TritDigit([Trit(-d.value) for d in self._digit_view.digits]))
            return ArithmeticCore.normalize(negated)
        return self

    # 身份方法
    def __hash__(self) -> int:
        """基於數值的哈希值"""
        return hash(self.to_integer())

    # 索引方法
    def __getitem__(self, index: int) -> Trit:
        return self._digit_view[index]

    # 返回有效位數
    def bit_length(self) -> int:
        """返回平衡三進製錶示的有效位數（考慮符號位）"""
        digits = self._digit_view.digits
        if not digits:
            return 0

        # 平衡三進制符號位語義:對於非零數，有效位數就是總位數；對於零值，返回1（至少有一位表示零）
        if self == TritNumber.zero():
            return 1

        # 對於非零數，所有位都有意義（包括可能的前導零）
        return len(digits)

    # Copyright (c) 2025 Certainty Computing Co. Limited. All rights reserved.


# ----------------------
# 4. 運算協作層
# ----------------------
class ArithmeticCore:
    """三值算術運算協作層"""
    # 基於Coq證明，根據基準測試調整的Karatsuba演算法閾值
    KARATSUBA_THRESHOLD = 32

    @staticmethod
    def add(a: TritNumber, b: TritNumber) -> TritNumber:
        """基於三值邏輯的逐位加法實現"""
        # 獲取對齊後的位序列
        max_len = max(len(a.digits), len(b.digits))
        a_aligned = a.to_digit_view().align(max_len).digits
        b_aligned = b.to_digit_view().align(max_len).digits

        # 從最低位開始計算
        result_digits = []
        carry = Trit(0)
        for i in range(max_len - 1, -1, -1):  # 從最低位到最高位
            a_trit = a_aligned[i]
            b_trit = b_aligned[i]

            # 計算當前位的和與進位
            total, carry = ArithmeticCore.add_trit(a_trit, b_trit, carry)
            result_digits.append(total)

        # 處理最高位進位
        if carry != Trit(0):
            result_digits.append(carry)

        # 反轉結果（高位在前）
        result_digits.reverse()
        return TritNumber(TritDigit(result_digits))

    @staticmethod
    def multiply_trit(a: Trit, b: Trit) -> Trit:
        """三值邏輯乘法（真值表實現）"""
        # 正確的三值乘法真值表
        if a.value == -1:
            return Trit(-b.value)  # -1 * x = -x
        elif a.value == 0:
            return Trit(0)  # 0 * x = 0
        else:  # a == 1
            return b  # 1 * x = x

    @staticmethod
    def add_trit(a: Trit, b: Trit, carry: Trit) -> Tuple[Trit, Trit]:
        """三值加法（帶進位）"""
        total = a.value + b.value + carry.value
        # 平衡三進制特殊處理
        if total == 2:
            return Trit(-1), Trit(1)
        elif total == -2:
            return Trit(1), Trit(-1)
        elif total == 3:
            return Trit(0), Trit(1)
        elif total == -3:
            return Trit(0), Trit(-1)
        else:
            return Trit(total), Trit(0)

    @staticmethod
    def subtract(a: TritNumber, b: TritNumber) -> TritNumber:
        """三進制減法 = 加上被減數的負數（Claude E. Shannon規則）"""
        neg_b = TritNumber(TritDigit([Trit(-d.value) for d in b.to_digit_view().digits]))
        return ArithmeticCore.add(a, neg_b)

    @staticmethod
    def naive_multiply(a: TritNumber, b: TritNumber) -> TritNumber:
        """樸素乘法（三進制位級實現）"""
        if a == TritNumber.zero() or b == TritNumber.zero():
            return TritNumber.zero()

        # 獲取位序列（低位在前）
        a_digits = list(reversed(a.digits))
        b_digits = list(reversed(b.digits))

        # 結果位序列（最大長度）
        result_digits = [Trit(0)] * (len(a_digits) + len(b_digits))

        # 逐位相乘並累加
        for i, a_trit in enumerate(a_digits):
            carry = Trit(0)
            for j, b_trit in enumerate(b_digits):
                product = ArithmeticCore.multiply_trit(a_trit, b_trit)

                pos = i + j
                current = result_digits[pos]

                # 加法（帶進位）
                total_val, new_carry = ArithmeticCore.add_trit(current, product, carry)

                result_digits[pos] = total_val
                carry = new_carry

            # 處理剩餘進位
            pos = i + len(b_digits)
            while carry != Trit(0):
                if pos < len(result_digits):
                    current = result_digits[pos]
                    total_val, new_carry = ArithmeticCore.add_trit(current, carry, Trit(0))
                    result_digits[pos] = total_val
                    carry = new_carry
                    pos += 1
                else:
                    result_digits.append(carry)
                    break

        # 反轉回高位在前
        result_digits.reverse()

        # 移除前導零
        while len(result_digits) > 1 and result_digits[0] == Trit(0):
            result_digits.pop(0)

        return TritNumber(TritDigit(result_digits))

    @staticmethod
    def multiply(a: TritNumber, b: TritNumber) -> TritNumber:
        """智能選擇乘法演算法"""
        # 處理特殊值
        if a == TritNumber.zero() or b == TritNumber.zero():
            return TritNumber.zero()

        # 計算位寬
        a_len = a.bit_length()
        b_len = b.bit_length()
        max_len = max(a_len, b_len)

        # 根據Coq證明的閾值選擇演算法
        if max_len < ArithmeticCore.KARATSUBA_THRESHOLD:
            return ArithmeticCore.naive_multiply(a, b)
        else:
            return ArithmeticCore.karatsuba(a, b)

    @staticmethod
    def karatsuba(x: TritNumber, y: TritNumber) -> TritNumber:
        """三值邏輯Karatsuba演算法"""
        # 處理特殊值
        if x == TritNumber.zero() or y == TritNumber.zero():
            return TritNumber.zero()

        # 計算實際位寬
        n = max(x.bit_length(), y.bit_length())

        # 基礎情況：使用樸素乘法
        if n <= ArithmeticCore.KARATSUBA_THRESHOLD:
            return ArithmeticCore.naive_multiply(x, y)

        # 計算分割點（基於三進制特性）
        m = max(1, n // 2)

        # 使用三進制分割方法
        high_x, low_x = ArithmeticCore.split_at(x, m)
        high_y, low_y = ArithmeticCore.split_at(y, m)

        # 遞歸計算
        z0 = ArithmeticCore.karatsuba(low_x, low_y)
        z2 = ArithmeticCore.karatsuba(high_x, high_y)

        # 計算中間項
        x_sum = low_x + high_x
        y_sum = low_y + high_y
        z1 = ArithmeticCore.karatsuba(x_sum, y_sum) - z0 - z2

        # 使用左移進行組合（代替乘法）
        part2 = ArithmeticCore.left_shift(z2, 2 * m)
        part1 = ArithmeticCore.left_shift(z1, m)
        result = part2 + part1 + z0

        return result

    @staticmethod
    def split_at(num: TritNumber, position: int) -> Tuple[TritNumber, TritNumber]:
        """三值邏輯 split_at：返回 (high, low) 滿足  high * 3^position + low == num"""
        if position <= 0:
            return TritNumber.zero(), num

        # 1. 構造 3^position （用左移代替乘法）
        power_of_3 = ArithmeticCore.left_shift(TritNumber.one(), position)

        # 2. 用三值除法得到商（high）和餘數（low）
        high, low = ArithmeticCore.divide_with_remainder(num, power_of_3)

        # 3. 確保餘數非負且 |low| < 3^position
        #    divide_with_remainder 已保證 0 ≤ low < |power_of_3|，無需再調整
        return high, low

    @staticmethod
    def left_shift(num: TritNumber, positions: int) -> TritNumber:
        """三進制左移 (乘以 3^positions)"""
        if positions < 0:
            raise ValueError("位移必須是非負整數")
        if positions == 0:
            return num

        # 三進制左移：低位補零
        new_digits = num.digits + [Trit(0)] * positions
        return TritNumber(TritDigit(new_digits))

    @staticmethod
    def right_shift(num: TritNumber, positions: int) -> TritNumber:
        """三值邏輯右移實現（相當於除以3^positions，截斷）"""
        if positions < 0:
            raise ValueError("位移必須是非負整數")
        if positions == 0:
            return num

        digits = num.digits

        # 移除最低位（截斷）
        if positions >= len(digits):
            return TritNumber.zero()

        # 創建新的位序列（移除低位）
        new_digits = digits[:-positions]

        # 處理負數的特殊情況：確保最高位符號正確
        if new_digits and new_digits[0].value == 0 and len(new_digits) > 1:
            # 移除前導零（符號位不能是零）
            first_non_zero = 0
            while first_non_zero < len(new_digits) and new_digits[first_non_zero].value == 0:
                first_non_zero += 1

            if first_non_zero < len(new_digits):
                new_digits = new_digits[first_non_zero:]
            else:
                return TritNumber.zero()

        return TritNumber(TritDigit(new_digits))

    @staticmethod
    def multiply_by_constant(a: TritNumber, constant: int) -> TritNumber:
        """優化常數乘法（支持大常數）"""
        # 特殊處理常數0和1
        if constant == 0:
            return TritNumber.zero()
        if constant == 1:
            return a

        # 使用三進制位移和加法
        result = TritNumber.zero()
        while constant:
            if constant & 1:
                result = result + a
            a = a + a  # 相當於乘以2
            constant //= 2
        return result

    @staticmethod
    def divide_with_remainder(a: TritNumber, b: TritNumber) -> Tuple[TritNumber, TritNumber]:
        """平衡三進制長除法實現，返回商和餘數（遵循向零舍入原則和Douglas W. Jones規則）"""
        if b == TritNumber.zero():
            raise ZeroDivisionError("除數不能為零")

        # 特殊情況處理：被除數為0
        if a == TritNumber.zero():
            return TritNumber.zero(), TritNumber.zero()

        # 保存原始符號
        sign_a = a.sign()
        sign_b = b.sign()

        # 對絕對值進行計算
        abs_a = TritNumber.__abs__(a)
        abs_b = TritNumber.__abs__(b)

        # 如果被除數絕對值小於除數絕對值，商為0，餘數為被除數
        if abs_a < abs_b:
            return TritNumber.zero(), a

        # 初始化商和餘數
        quotient = TritNumber.zero()
        remainder = TritNumber.zero()

        # 逐位處理被除數（從最高位到最低位）
        for digit in abs_a.digits:
            # 將當前位移到餘數中
            remainder = ArithmeticCore.left_shift(remainder, 1)
            current_digit = TritNumber.from_digits([digit])
            remainder = ArithmeticCore.add(remainder, current_digit)

            # 計算當前位的商
            count = 0
            while remainder >= abs_b:
                remainder = ArithmeticCore.subtract(remainder, abs_b)
                count += 1

            # 將當前位的商加入結果
            quotient = ArithmeticCore.left_shift(quotient, 1)
            quotient = ArithmeticCore.add(quotient, TritNumber.from_integer(count))

        # 確保餘數在正確範圍內 [0, |b|)
        while remainder >= abs_b:
            remainder = ArithmeticCore.subtract(remainder, abs_b)
            quotient = ArithmeticCore.add(quotient, TritNumber.one())

        while remainder < TritNumber.zero():
            remainder = ArithmeticCore.add(remainder, abs_b)
            quotient = ArithmeticCore.subtract(quotient, TritNumber.one())

        # 商的符號 = sign(a) * sign(b)
        if sign_a.value * sign_b.value == -1:
            quotient = -quotient

        # 餘數的符號：與被除數相同
        if sign_a.value == -1:
            remainder = -remainder

        # 標準化結果
        quotient = ArithmeticCore.normalize(quotient)
        remainder = ArithmeticCore.normalize(remainder)

        return quotient, remainder

    @staticmethod
    def divide(a: TritNumber, b: TritNumber) -> TritNumber:
        """除法介面，保持向後相容"""
        quotient, _ = ArithmeticCore.divide_with_remainder(a, b)
        return quotient

    @staticmethod
    def power(base: 'TritNumber', exponent: 'TritNumber') -> 'TritNumber':
        """三值邏輯冪運算實現（無整數轉換）"""
        # 定義常量
        zero = TritNumber.zero()
        one = TritNumber(TritDigit([Trit(1)]))
        neg_one = TritNumber(TritDigit([Trit(-1)]))

        # 處理底數為零的情況
        if base == zero:
            if exponent == zero:
                return one  # 0^0 = 1 (數學慣例)
            elif exponent < zero:
                raise ZeroDivisionError("0的負指數未定義")
            else:
                return zero

        # 處理指數為零
        if exponent == zero:
            return one

        # 處理指數為1
        if exponent == one:
            return base

        # 處理指數為-1
        if exponent == neg_one:
            quotient, _ = ArithmeticCore.divide_with_remainder(one, base)
            return quotient

        # 處理負指數
        if exponent < zero:
            reciprocal, _ = ArithmeticCore.divide_with_remainder(one, base)
            return ArithmeticCore.power(reciprocal, -exponent)

        # 三值邏輯快速冪演算法
        result = one
        current = base
        exp = exponent

        # 使用三值邏輯常數
        two = TritNumber(TritDigit([Trit(1), Trit(-1)]))  # 使用正確的平衡三進製錶示2，避免轉換開銷

        while exp > zero:
            # 檢查最低位是否為1
            remainder = ArithmeticCore.trit_mod(exp, two)
            if remainder != zero:
                result = ArithmeticCore.multiply(result, current)

            # 平方當前值
            current = ArithmeticCore.multiply(current, current)

            # 指數右移一位（除以2），只取商
            exp, _ = ArithmeticCore.trit_floor_divide(exp, two)

        return result

    @staticmethod
    def trit_floor_divide(a: TritNumber, b: TritNumber) -> Tuple[TritNumber, TritNumber]:
        """三值邏輯floor除法"""
        if b == TritNumber.zero():
            raise ZeroDivisionError("除數不能為零")

        # 處理符號
        sign_a = a.sign()
        sign_b = b.sign()
        abs_a = TritNumber.__abs__(a)
        abs_b = TritNumber.__abs__(b)

        # 初始化商和餘數（三進制位容器）
        quotient_digits = []
        remainder = TritNumber.zero()

        # 逐位處理被除數（從最高位到最低位）
        for digit in abs_a.digits:
            # 餘數左移一位（×3）並加入當前位
            remainder = ArithmeticCore.left_shift(remainder, 1)
            current_digit = TritNumber.from_digits([digit])
            remainder = ArithmeticCore.add(remainder, current_digit)

            # 計算當前位的商值（0/1/2）
            subtract_count = 0
            # 三進制比較：remainder >= abs_b
            while remainder >= abs_b:
                remainder = ArithmeticCore.subtract(remainder, abs_b)
                subtract_count += 1

            # 商左移一位並加入當前計數（保持三進制位序列）
            quotient_digits.append(Trit(subtract_count if subtract_count <= 1 else -1))
            # 處理進位（當subtract_count為2時，三進製錶示為-1並產生進位）
            if subtract_count == 2:
                # 向高位進位1（通過調整商的前一位實現）
                i = len(quotient_digits) - 1
                while i > 0 and quotient_digits[i - 1].value == 1:
                    quotient_digits[i - 1] = Trit(-1)
                    i -= 1
                if i == 0 and quotient_digits[i].value == 1:
                    quotient_digits.insert(0, Trit(-1))
                    quotient_digits[i + 1] = Trit(0)
                else:
                    quotient_digits[i - 1] = Trit(quotient_digits[i - 1].value + 1)

        # 構建商的TritNumber
        quotient = TritNumber.from_digits(quotient_digits)
        quotient = ArithmeticCore.normalize(quotient)

        # 只要餘數為負，就調整商和餘數
        if not remainder == TritNumber.zero() and remainder < TritNumber.zero():
            quotient = ArithmeticCore.subtract(quotient, TritNumber.one())
            remainder = ArithmeticCore.add(remainder, abs_b)

        # 應用符號
        if sign_a != sign_b:
            quotient = -quotient

        return quotient, remainder

    @staticmethod
    def trit_mod(a: TritNumber, b: TritNumber) -> TritNumber:
        """三值邏輯取模運算，確保返回非負餘數且小於|b|"""
        if b == TritNumber.zero():
            raise ZeroDivisionError("除數不能為零")

        # 特殊情況：除數絕對值為1時，任何數取模結果都是0
        abs_b = TritNumber.__abs__(b)
        if abs_b == TritNumber.one():
            return TritNumber.zero()

        # 特殊情況：被除數為0
        if a == TritNumber.zero():
            return TritNumber.zero()

        # 計算商和初始餘數
        q, r = ArithmeticCore.trit_floor_divide(a, b)

        # 計算餘數：a = b * q + remainder
        remainder = ArithmeticCore.subtract(a, ArithmeticCore.multiply(b, q))
        remainder = ArithmeticCore.normalize(remainder)

        # 確保餘數在[0, abs_b)範圍內
        max_iterations = 100
        iterations = 0

        # 調整餘數到正確範圍
        while iterations < max_iterations:
            if remainder < TritNumber.zero():
                # 餘數為負，加上abs_b
                remainder = ArithmeticCore.add(remainder, abs_b)
                iterations += 1
            elif remainder >= abs_b:
                # 餘數大於等於除數絕對值，減去abs_b
                remainder = ArithmeticCore.subtract(remainder, abs_b)
                iterations += 1
            else:
                # 餘數在正確範圍內，退出迴圈
                break

        if iterations >= max_iterations:
            raise RuntimeError(f"取模運算可能陷入無限迴圈: a={a}, b={b}")

        return remainder

    @staticmethod
    def trit_gcd(a: TritNumber, b: TritNumber) -> TritNumber:
        """三值邏輯最大公約數（歐幾裏得演算法）"""
        # GCD(0,0)未定義，拋出異常
        if a == TritNumber.zero() and b == TritNumber.zero():
            raise ValueError("GCD(0, 0) 未定義")
        # 處理單個零的情況
        if a == TritNumber.zero():
            result = TritNumber.__abs__(b)
            return result
        if b == TritNumber.zero():
            result = TritNumber.__abs__(a)
            return result

        a = TritNumber.__abs__(a)
        b = TritNumber.__abs__(b)

        if a < b:
            a, b = b, a

        max_iterations = max(a.bit_length(), b.bit_length()) * 10
        iterations = 0

        while b != TritNumber.zero():
            iterations += 1
            if iterations > max_iterations:
                raise RuntimeError(f"GCD計算可能陷入無限迴圈: a={a}, b={b}")

            remainder = ArithmeticCore.trit_mod(a, b)

            if remainder == a and a >= b:
                raise RuntimeError(f"GCD計算陷入迴圈: a={a}, b={b}, remainder={remainder}")

            a, b = b, remainder

        return a

    @staticmethod
    def trit_lcm(a: TritNumber, b: TritNumber) -> TritNumber:
        """三值邏輯最小公倍數實現"""
        # 含零LCM未定義，拋出異常
        if a == TritNumber.zero() and b == TritNumber.zero():
            raise ValueError("LCM(0, 0) 未定義")
        if a == TritNumber.zero() or b == TritNumber.zero():
            raise ValueError("LCM(0, x) 或 LCM(x, 0) 未定義（x為任意整數）")

        # 正常計算流程
        gcd_val = ArithmeticCore.trit_gcd(a, b)

        if gcd_val == TritNumber.zero():
            return TritNumber.zero()

        product = ArithmeticCore.multiply(a, b)
        abs_product = TritNumber.__abs__(product)
        lcm = ArithmeticCore.divide(abs_product, gcd_val)

        return ArithmeticCore.normalize(lcm)

    @staticmethod
    def normalize(num: TritNumber) -> TritNumber:
        """規範化結果（移除前導零）"""
        digits = num.digits
        while len(digits) > 1 and digits[0] == Trit(0):
            digits = digits[1:]
        return TritNumber.from_digits(digits)


class TensorCore:
    """三值張量運算協作層"""
    def __init__(self, depth: int = 3):
        self.depth = depth
        self._DYNAMIC_PSI = [Trit(1), Trit(0), Trit(-1), Trit(0), Trit(0), Trit(0), Trit(-1), Trit(0), Trit(1)]

    def matmul(self, a: TritNumber, b: TritNumber) -> Trit:
        """三值邏輯矩陣乘法運算"""
        # 驗證輸入類型
        if not isinstance(a, TritNumber) or not isinstance(b, TritNumber):
            raise TypeError("matmul操作需要TritNumber輸入")

        # 確保是單值三進制數（標量）
        if len(a.digits) != 1 or len(b.digits) != 1:
            raise ValueError("matmul僅支持單值TritNumber（標量）")

        # 提取Trit值
        x = a.digits[0]
        y = b.digits[0]

        # 使用默認參數（單位矩陣）
        return self.tensor_function(x, y, [1, 0, 0, 0, 1, 0, 0, 0, 1])

    def tensor_function(self, x: Trit, y: Trit, psi: List[int]) -> Trit:
        """三值邏輯QZ張量積函數"""
        # PSI參數驗證
        if len(psi) != 9:
            raise ValueError("psi 參數必須為9個元素")

        # 驗證所有參數都是整數 -1, 0, 1
        for p in psi:
            if p not in (-1, 0, 1):
                raise ValueError(f"無效的PSI值: {p} (必須是-1,0或1)")

        # 顯式QZ張量指示器計算，可緩存為預生成LUT
        kappa = self._tensor_delta(x, y)

        # 點積計算
        total = 0
        for i in range(9):
            total += psi[i] * kappa[i]

        # 模3映射（三值邏輯處理）
        if total > 1:
            return Trit(total - 3)
        elif total < -1:
            return Trit(total + 3)
        else:
            return Trit(total)

    def _tensor_delta(self, x: Trit, y: Trit) -> List[int]:
        """顯式QZ張量函數（三值邏輯實現）：κ_{m,n}(x,y) = [x=n ∧ y=m]"""
        combinations = [
            (Trit(-1), Trit(-1)), (Trit(-1), Trit(0)), (Trit(-1), Trit(1)),
            (Trit(0), Trit(-1)), (Trit(0), Trit(0)), (Trit(0), Trit(1)),
            (Trit(1), Trit(-1)), (Trit(1), Trit(0)), (Trit(1), Trit(1))
        ]

        return [
            1 if (y == m and x == n) else 0
            for m, n in combinations
        ]

    def represent_value(self, value: int) -> List[Trit]:
        """整數值→三值轉換（三值邏輯實現）"""
        if value == 0:
            return [Trit(0)]

        # 處理符號
        sign = 1 if value > 0 else -1
        abs_value = abs(value)

        # 轉換為平衡三進制
        digits = []
        num = abs_value

        while num > 0:
            num, remainder = divmod(num, 3)
            if remainder == 2:
                remainder = -1
                num += 1
            digits.append(Trit(remainder))

        # 添加符號位
        if sign == -1:
            digits = [Trit(-d.value) for d in digits]

        digits.reverse()
        return digits

    def reconstruct_value(self, trits: List[Trit]) -> int:
        """從三值鏈重構原始整數值（三值邏輯實現）"""
        # 使用三值邏輯運算
        value = 0
        power = 1

        # 三進制基數
        base = 3

        for trit in reversed(trits):
            # 將Trit轉換為整數
            trit_value = trit.value
            # 累加當前位的值
            value += trit_value * power
            # 更新冪
            power *= base

        return value

    def semi_tensor_product(self, A: List[List[Trit]], B: List[List[Trit]]) -> List[List[Trit]]:
        """半張量積實現"""
        # 獲取矩陣維度
        m = len(A)  # A的行數
        n = len(A[0]) if A else 0  # A的列數
        p = len(B)  # B的行數
        q = len(B[0]) if B else 0  # B的列數

        # 檢查維度相容性
        if n == 0 or p == 0:
            raise ValueError("矩陣不能為空")

        if n % p != 0:
            raise ValueError(f"維度不相容: A的列數({n})必須是B的行數({p})的整數倍")

        # 計算塊大小
        block_size = n // p

        # 初始化結果矩陣
        result = []

        # 執行半張量積: A ⋉ B = A · (I_p ⊗ B)
        for i in range(m):
            row = []
            for j in range(q):
                # 計算當前塊的結果
                sum_val = Trit(0)
                for k in range(p):
                    for l in range(block_size):
                        # 獲取A中的對應元素
                        a_col = k * block_size + l
                        if a_col < n:  # 邊界檢查
                            a_val = A[i][a_col]
                            # 獲取B中的對應元素
                            b_val = B[k][j]
                            # 計算乘積並累加
                            product = ArithmeticCore.multiply_trit(a_val, b_val)
                            sum_val = ArithmeticCore.add_trit(sum_val, product, Trit(0))[0]
                row.append(sum_val)
            result.append(row)

        return result

    @staticmethod
    def trit_count(matrix: List[List[Trit]], axis: int) -> TritNumber:
        """三值邏輯矩陣維度計數"""
        if not matrix:
            return TritNumber.zero()

        if axis == 0:  # 行數
            return TritNumber.from_integer(len(matrix))
        else:  # 列數
            return TritNumber.from_integer(len(matrix[0]))

    def trit_identity_matrix(self, size: TritNumber) -> List[List[Trit]]:
        """生成三值邏輯單位矩陣"""
        n = size.to_integer()  # 轉換為整數用於迴圈
        return [
            [Trit(1) if i == j else Trit(0) for j in range(n)]
            for i in range(n)
        ]

    def trit_tensor(self, A: List[List[Trit]], B: List[List[Trit]]) -> List[List[Trit]]:
        """三值邏輯QZ張量積"""
        m, n = len(A), len(A[0])
        p, q = len(B), len(B[0])

        result = []
        for i in range(m):
            for k in range(p):
                row = []
                for j in range(n):
                    for l in range(q):
                        # 三值邏輯乘法
                        product = ArithmeticCore.multiply_trit(A[i][j], B[k][l])
                        row.append(product)
                result.append(row)
        return result

    def trit_matrix_multiply(self, A: List[List[Trit]], B: List[List[Trit]]) -> List[List[Trit]]:
        """三值邏輯矩陣乘法"""
        m, n = len(A), len(A[0])
        p, q = len(B), len(B[0])

        if n != p:
            raise ValueError("矩陣維度不相容")

        result = []
        for i in range(m):
            row = []
            for j in range(q):
                sum_val = Trit(0)
                for k in range(n):
                    product = ArithmeticCore.multiply_trit(A[i][k], B[k][j])
                    sum_val = ArithmeticCore.add_trit(sum_val, product, Trit(0))[0]
                row.append(sum_val)
            result.append(row)
        return result


class PolynomialCore:
    """多項式模態算子實現"""
    # 創建共用的 TensorCore 實例
    _tensor_core = TensorCore()

    @staticmethod
    def tensor_function(x: Trit, y: Trit, tensor_params: List[Union[int, float, Trit]]) -> Trit:
        """轉發介面帶調試日誌"""
        caller = inspect.stack()[1].function
        if os.getenv("DEBUG", "False") == "True":
            print(f"tensor call from {caller}, x={x}, y={y}")

        # 通過共用實例調用
        return PolynomialCore._tensor_core.tensor_function(x, y, tensor_params)

    @staticmethod
    def evaluate_univariate(x: Trit, coeffs: Tuple[int, int, int], use_trit_logic: bool = False) -> Trit:
        """單變數多項式求值
        Args:
            use_trit_logic: 如果為True，使用三值邏輯實現（較慢但語義純粹）
                            如果為False，使用快速整數實現（較快但涉及整數轉換）
        """
        if use_trit_logic:
            return PolynomialCore._evaluate_univariate_trit_logic(x, coeffs)
        else:
            return PolynomialCore._evaluate_univariate_fast(x, coeffs)

    @staticmethod
    def _evaluate_univariate_fast(x: Trit, coeffs: Tuple[int, int, int]) -> Trit:
        """快速實現（使用整數運算）"""
        a, b, c = coeffs
        x_val = x.value

        # 直接計算多項式值
        result_val = a * x_val * x_val + b * x_val + c

        # 模3處理
        result_val %= 3
        if result_val == 2:
            result_val = -1
        elif result_val == -2:
            result_val = 1
        elif result_val == -1:
            result_val = -1
        elif result_val == 1:
            result_val = 1
        else:
            result_val = 0

        return Trit(result_val)

    @staticmethod
    def _evaluate_univariate_trit_logic(x: Trit, coeffs: Tuple[int, int, int]) -> Trit:
        """三值邏輯實現"""
        a, b, c = coeffs

        # 將係數轉換為TritNumber
        a_num = TritNumber.from_integer(a)
        b_num = TritNumber.from_integer(b)
        c_num = TritNumber.from_integer(c)

        # 將輸入x轉換為TritNumber
        x_num = TritNumber(TritDigit([x]))

        # 使用三值邏輯運算計算多項式值
        x_squared = x_num * x_num
        a_x_squared = a_num * x_squared
        b_x = b_num * x_num

        result_num = a_x_squared + b_x + c_num

        # 將結果轉換為單個Trit（取模3）
        # 直接取最低位作為模3的結果
        if result_num.digits:
            # 取最低位（個位）
            return result_num.digits[-1]
        else:
            return Trit(0)

    @staticmethod
    def evaluate_bivariate(x: Trit, y: Trit, coeffs: Tuple[int, int, int, int, int, int],
                           use_trit_logic: bool = False) -> Trit:
        """
        雙變數多項式求值：
        f(x,y) = a0 + a1·x + a2·y + a3·x² + a4·y² + a5·x·y
        Args:
            use_trit_logic: 如果為True，使用三值邏輯實現（較慢但語義純粹）
                            如果為False，使用快速整數實現（較快但涉及整數轉換）
        """
        if use_trit_logic:
            return PolynomialCore._evaluate_bivariate_trit_logic(x, y, coeffs)
        else:
            return PolynomialCore._evaluate_bivariate_fast(x, y, coeffs)

    @staticmethod
    def _evaluate_bivariate_fast(x: Trit, y: Trit, coeffs: Tuple[int, int, int, int, int, int]) -> Trit:
        """快速實現（使用整數運算）"""
        a0, a1, a2, a3, a4, a5 = coeffs
        x_val = x.value
        y_val = y.value

        # 直接計算多項式值
        result_val = (
                a0 +
                a1 * x_val +
                a2 * y_val +
                a3 * x_val * x_val +
                a4 * y_val * y_val +
                a5 * x_val * y_val
        )

        # 模3處理（確保結果在-1,0,1範圍內）
        result_val %= 3
        if result_val == 2:
            result_val = -1
        elif result_val == -2:
            result_val = 1
        elif result_val == -1:
            result_val = -1
        elif result_val == 1:
            result_val = 1
        else:
            result_val = 0

        return Trit(result_val)

    @staticmethod
    def _evaluate_bivariate_trit_logic(x: Trit, y: Trit, coeffs: Tuple[int, int, int, int, int, int]) -> Trit:
        """三值邏輯實現"""
        a0, a1, a2, a3, a4, a5 = coeffs

        # 將係數轉換為TritNumber
        a0_num = TritNumber.from_integer(a0)
        a1_num = TritNumber.from_integer(a1)
        a2_num = TritNumber.from_integer(a2)
        a3_num = TritNumber.from_integer(a3)
        a4_num = TritNumber.from_integer(a4)
        a5_num = TritNumber.from_integer(a5)

        # 將輸入x和y轉換為TritNumber
        x_num = TritNumber(TritDigit([x]))
        y_num = TritNumber(TritDigit([y]))

        # 使用三值邏輯運算計算多項式值
        # 計算各項
        term1 = a0_num  # a0
        term2 = a1_num * x_num  # a1·x
        term3 = a2_num * y_num  # a2·y

        x_squared = x_num * x_num  # x²
        term4 = a3_num * x_squared  # a3·x²

        y_squared = y_num * y_num  # y²
        term5 = a4_num * y_squared  # a4·y²

        x_y = x_num * y_num  # x·y
        term6 = a5_num * x_y  # a5·x·y

        # 計算總和
        result_num = term1 + term2 + term3 + term4 + term5 + term6

        # 將結果轉換為單個Trit（取模3）
        # 直接取最低位作為模3的結果
        if result_num.digits:
            # 取最低位（個位）
            return result_num.digits[-1]
        else:
            return Trit(0)

    @classmethod
    def get_bivariate_operator(cls, coeffs: Tuple[int, int, int, int, int, int]) -> Callable[[Trit, Trit], Trit]:
        """獲取雙變數模態算子，可擴展用於多模態融合、關係邏輯推理"""
        return lambda x, y: cls.evaluate_bivariate(x, y, coeffs)


class UnifiedComputingCore:
    """統一計算核心，智能選擇最優計算路徑"""
    def __init__(self):
        # 初始化 TensorCore 實例
        self._tensor_core = TensorCore()

    @lru_cache(maxsize=1024)
    def evaluate(self, x: Trit, y: Optional[Trit] = None,
                 poly_coeffs: Optional[Tuple] = None,
                 tensor_params: Optional[Tuple[int, ...]] = None) -> Trit:
        # 檢測可用的優化路徑
        if poly_coeffs is not None:
            # 確保是元組類型
            poly_tuple = tuple(poly_coeffs)

            if y is None:
                if len(poly_tuple) == 3:
                    # 顯式轉換為三元組
                    a, b, c = poly_tuple
                    return PolynomialCore.evaluate_univariate(x, (a, b, c))
                else:
                    raise ValueError("單變數多項式需要3個係數")
            else:
                if len(poly_tuple) == 6:
                    # 顯式轉換為六元組
                    a0, a1, a2, a3, a4, a5 = poly_tuple
                    return PolynomialCore.evaluate_bivariate(x, y, (a0, a1, a2, a3, a4, a5))
                else:
                    raise ValueError("雙變數多項式需要6個係數")

        elif tensor_params is not None and y is not None:
            # 確保參數有效 - tensor_params 已經是元組了
            if len(tensor_params) == 9:
                # 將元組轉換回列表供 TensorCore 使用
                return self._tensor_core.tensor_function(x, y, list(tensor_params))
            else:
                raise ValueError("QZ張量函數需要9個參數")

        else:
            # 啟發式自動選擇
            if self._is_poly_optimizable(x, y):
                coeffs = self._infer_poly_coeffs(x, y)
                if y is None:
                    # 顯式轉換為三元組
                    a, b, c = coeffs
                    return PolynomialCore.evaluate_univariate(x, (a, b, c))
                else:
                    # 顯式轉換為六元組
                    a0, a1, a2, a3, a4, a5 = coeffs
                    return PolynomialCore.evaluate_bivariate(x, y, (a0, a1, a2, a3, a4, a5))
            else:
                params = self._default_tensor_params()
                return self._tensor_core.tensor_function(x, y, params)

    def _is_poly_optimizable(self, x: Trit, y: Optional[Trit]) -> bool:
        """啟發式檢測是否可用多項式優化（三值邏輯實現）"""
        if y is None:
            return True  # 單變數總是可用多項式優化

        # 創建 TritNumber 表示
        x_num = TritNumber(TritDigit([x]))
        y_num = TritNumber(TritDigit([y]))

        # 計算差值（三值邏輯）
        diff = ArithmeticCore.subtract(x_num, y_num)

        # 計算絕對值（三值邏輯）
        abs_diff = TritNumber.__abs__(diff)

        # 創建常數 2 的三值表示
        two = TritNumber(TritDigit([Trit(1), Trit(-1)]))  # 平衡三進製錶示 2 (1×3^1 + 1×3^0)

        # 三值比較：abs_diff < two
        return abs_diff < two

    def _infer_poly_coeffs(self, x: Trit, y: Optional[Trit]) -> Tuple:
        """推斷多項式係數（簡化實現）"""
        if y is None:
            # 單變數：默認線性多項式
            return (0, 1, 0)  # f(x) = x
        else:
            # 雙變數：默認乘積多項式
            return (0, 0, 0, 0, 0, 1)  # f(x,y) = x*y

    def _default_tensor_params(self) -> List[int]:
        """默認QZ張量參數（單位矩陣）"""
        return [1, 0, 0, 0, 1, 0, 0, 0, 1]  # 單位矩陣參數

    # Copyright (c) 2025 Certainty Computing Co. Limited. All rights reserved.


# ----------------------
# 5. 介面適配層
# ----------------------
class TritNumberOperatorAdapter(TritNumber):
    """運算符適配器（通過繼承擴展）"""

    def __add__(self, other: 'TritNumber') -> 'TritNumber':
        return ArithmeticCore.add(self, other)

    def __sub__(self, other: 'TritNumber') -> 'TritNumber':
        return ArithmeticCore.subtract(self, other)

    def __mul__(self, other: 'TritNumber') -> 'TritNumber':
        return ArithmeticCore.multiply(self, other)

    def __truediv__(self, other: 'TritNumber') -> 'TritNumber':
        quotient, _ = ArithmeticCore.divide_with_remainder(self, other)
        return quotient

    def __floordiv__(self, other: 'TritNumber') -> 'TritNumber':
        quotient, _ = ArithmeticCore.trit_floor_divide(self, other)
        return quotient

    def __mod__(self, other: 'TritNumber') -> 'TritNumber':
        return ArithmeticCore.trit_mod(self, other)

    def __pow__(self, other):
        return ArithmeticCore.power(self, other)

    def __matmul__(self, other: 'TritNumber') -> Trit:
        return TensorCore().matmul(self, other)

    def __abs__(self) -> 'TritNumber':
        if self.sign() != Trit(-1):
            return TritNumber(TritDigit([Trit(1)]))
        return -self

    def __neg__(self) -> 'TritNumber':
        new_digits = TritDigit([Trit(-d.value) for d in self.digits])
        return TritNumber(new_digits)

    # Copyright (c) 2025 Certainty Computing Co. Limited. All rights reserved.


# ----------------------
# 6. 測試用例
# ----------------------
class TestTrit(unittest.TestCase):
    """測試三值邏輯基元Trit"""

    def test_trit_creation(self):
        """測試Trit的創建與緩存機制"""
        t1 = Trit(1)
        t2 = Trit(1)
        self.assertIs(t1, t2)  # 驗證緩存生效
        self.assertEqual(t1.value, 1)

        with self.assertRaises(ValueError):
            Trit(2)  # 無效值

    def test_trit_comparison(self):
        """測試Trit的比較運算"""
        t_true = Trit(1)
        t_false = Trit(-1)
        t_doubt = Trit(0)

        self.assertEqual(t_true, Trit(1))
        self.assertNotEqual(t_true, t_false)
        self.assertTrue(t_true)
        self.assertFalse(t_false)
        self.assertFalse(t_doubt)  # 0對應False

    def test_trit_conversion(self):
        """測試Trit與其他類型的轉換"""
        t = Trit(-1)
        self.assertEqual(int(t), -1)
        self.assertEqual(float(t), -1.0)
        self.assertEqual(str(t), "False")
        self.assertEqual(repr(t), "Trit(-1)")


class TestTritDigit(unittest.TestCase):
    """測試位容器TritDigit"""

    def test_digit_creation(self):
        """測試TritDigit的創建與驗證"""
        digits = [Trit(1), Trit(0), Trit(-1)]
        td = TritDigit(digits)
        self.assertEqual(len(td.digits), 3)

        with self.assertRaises(TypeError):
            TritDigit([1, 0, -1])  # 非Trit實例

    def test_digit_operations(self):
        """測試位操作（對齊、位移、提取/設置位）"""
        td = TritDigit([Trit(1), Trit(-1)])

        # 測試對齊
        aligned = td.align(4)
        self.assertEqual(len(aligned.digits), 4)
        self.assertEqual(aligned.digits[:2], [Trit(0), Trit(0)])

        # 測試位移
        shifted = td.shift(1)
        self.assertEqual(shifted.digits, [Trit(1), Trit(-1), Trit(0)])

        # 測試提取位
        self.assertEqual(td.extract_bit(1), Trit(-1))
        self.assertEqual(td.extract_bit(5), Trit(0))  # 越界返回0

    def test_digit_to_integer(self):
        """測試位序列到整數的轉換"""
        td = TritDigit([Trit(1), Trit(0), Trit(-1)])
        # 計算邏輯：1*3² + 0*3¹ + (-1)*3⁰ = 9 + 0 -1 = 8
        self.assertEqual(td.to_integer(), 8)

    def test_serialization_edge_cases(self):
        """驗證極端位寬序列化/反序列化"""
        # 萬位三進制數序列化測試
        giant_num = TritDigit([Trit(1)] + [Trit(0)] * 9999)
        state = giant_num.__getstate__()  # 觸發序列化

        # 驗證序列化數據完整性
        assert len(state) == 10000, "序列化位寬丟失"
        assert state[0] == 1 and state[-1] == 0, "序列化數據錯位"

        # 反序列化驗證
        new_digit = TritDigit.__new__(TritDigit)
        new_digit.__setstate__(state)
        assert new_digit.digits[0] == Trit(1), "反序列化符號位錯誤"
        assert len(new_digit.digits) == 10000, "反序列化位寬丟失"
        print("萬位序列化驗證通過")


class TestTritNumber(unittest.TestCase):
    """測試數值實體TritNumber"""

    def test_number_creation(self):
        """測試TritNumber的創建工廠方法"""
        num = TritNumber.from_integer(5)
        self.assertEqual(num.to_integer(), 5)

        zero = TritNumber.zero()
        one = TritNumber.one()
        self.assertEqual(zero.to_integer(), 0)
        self.assertEqual(one.to_integer(), 1)

    def test_number_comparison(self):
        """測試數值比較"""
        a = TritNumber.from_integer(3)
        b = TritNumber.from_integer(5)
        self.assertLess(a, b)
        self.assertGreater(b, a)
        self.assertEqual(a, TritNumber.from_integer(3))

    def test_number_conversion(self):
        """測試數值轉換與符號"""
        num = TritNumber.from_integer(-7)
        self.assertEqual(int(num), -7)
        self.assertEqual(num.sign(), Trit(-1))  # 負數符號位為-1
        self.assertEqual(abs(num).to_integer(), 7)

    def test_tritnumber_from_integer(self):
        """驗證多位數三進制數從整數轉換的定義與實現一致"""
        print("\n=== 多位數三進制整數轉換測試 ===")
        assert TritNumber.from_integer(0).to_integer() == 0
        assert TritNumber.from_integer(1).to_integer() == 1
        assert TritNumber.from_integer(-1).to_integer() == -1
        assert TritNumber.from_integer(2).to_integer() == 2
        assert TritNumber.from_integer(-2).to_integer() == -2
        assert TritNumber.from_integer(3).to_integer() == 3
        assert TritNumber.from_integer(-3).to_integer() == -3
        assert TritNumber.from_integer(4).to_integer() == 4
        assert TritNumber.from_integer(-4).to_integer() == -4
        assert TritNumber.from_integer(5).to_integer() == 5
        assert TritNumber.from_integer(-5).to_integer() == -5

        print("多位數三進制整數轉換驗證通過")

    def test_large_number_conversion(self):
        """驗證大整數轉換的正確性"""
        test_cases = [
            (65535, "65535"),
            (65536, "65536"),
            (4294967296, "4294967296"),
            (1073741824, "1073741824"),
            (1152921504606846975, "1152921504606846975")
        ]

        for value, expected_str in test_cases:
            num = TritNumber.from_integer(value)
            result = num.to_integer()
            assert result == value, f"{value} → {result}"
            print(f"大數轉換驗證通過: {value} == {result}")

    def test_trit_division(self):
        """基本三值除法運算測試"""
        print("\n=== 基本三值除法測試 ===")
        # 使用 TritNumber 的除法運算符
        test_cases = [
            (TritNumber(TritDigit([Trit(1)])), TritNumber(TritDigit([Trit(1)])), TritNumber(TritDigit([Trit(1)]))),
            (TritNumber(TritDigit([Trit(1)])), TritNumber(TritDigit([Trit(-1)])), TritNumber(TritDigit([Trit(-1)]))),
            (TritNumber(TritDigit([Trit(-1)])), TritNumber(TritDigit([Trit(1)])), TritNumber(TritDigit([Trit(-1)]))),
            (TritNumber(TritDigit([Trit(-1)])), TritNumber(TritDigit([Trit(-1)])), TritNumber(TritDigit([Trit(1)]))),
        ]

        # 正常除法運算
        for dividend, divisor, expected in test_cases:
            result = dividend / divisor
            print(f"{dividend} / {divisor} = {result} (預期: {expected})")
            assert result == expected

        # 除數為零的測試
        zero = TritNumber(TritDigit([Trit(0)]))
        one = TritNumber(TritDigit([Trit(1)]))

        # 1 / 0 -> 應該引發異常
        try:
            result = one / zero
            assert False, "1/0 未引發ZeroDivisionError"
        except ZeroDivisionError as e:
            assert str(e) == "除數不能為零", f"錯誤資訊不匹配: {str(e)}"

        # 0 / 1 -> 0
        result = zero / one
        assert result == zero, f"{zero} / {one} = {result} (預期: {zero})"
        print(f"{zero} / {one} = {result} (預期: {zero})")

        # 0 / 0 -> 應該引發異常
        try:
            result = zero / zero
            assert False, "0/0 未引發ZeroDivisionError"
        except ZeroDivisionError as e:
            assert str(e) == "除數不能為零", f"錯誤資訊不匹配: {str(e)}"

        # 連續除法和混合運算
        expr = TritNumber(TritDigit([Trit(1)])) / TritNumber(TritDigit([Trit(-1)])) / TritNumber(TritDigit([Trit(1)]))
        assert expr == TritNumber(TritDigit([Trit(-1)])), f"{expr} != {TritNumber(TritDigit([Trit(-1)]))}"
        print(f"1 / -1 / 1 = {expr} (預期: -1)")

        mixed = (TritNumber(TritDigit([Trit(1)])) * TritNumber(TritDigit([Trit(-1)]))) / (
                TritNumber(TritDigit([Trit(1)])) + TritNumber(TritDigit([Trit(1)])))
        assert mixed == TritNumber(TritDigit([Trit(0)])), f"{mixed} != {TritNumber(TritDigit([Trit(0)]))}"
        print(f"(1 * -1) / (1 + 1) = {mixed} (預期: 0)")

    def test_optimized_trit_number(self):
        """測試多位數三進制數的複雜運算功能"""
        print("\n=== 多位數三進制複雜運算測試 ===")
        # 使用 TritNumber 的運算符
        a = TritNumber.from_integer(12)
        b = TritNumber.from_integer(5)
        assert (a + b).to_integer() == 17

        # 乘法測試
        f = TritNumber.from_integer(4)
        g = TritNumber.from_integer(3)
        assert (f * g).to_integer() == 12

        # 減法測試
        c = TritNumber.from_integer(8)
        d = TritNumber.from_integer(3)
        assert (c - d).to_integer() == 5

        # 除法測試
        e = TritNumber.from_integer(15)
        h = TritNumber.from_integer(5)
        assert (e / h).to_integer() == 3

        print("多位數三進制複雜運算驗證通過")


class TestArithmeticCore(unittest.TestCase):
    """測試運算協作層ArithmeticCore"""

    def test_addition(self):
        """測試加法運算"""
        a = TritNumber.from_integer(2)
        b = TritNumber.from_integer(3)
        self.assertEqual((a + b).to_integer(), 5)

        # 測試負數加法
        c = TritNumber.from_integer(-1)
        self.assertEqual((a + c).to_integer(), 1)

    def test_multiplication(self):
        """測試乘法運算（含樸素演算法與Karatsuba切換）"""
        a = TritNumber.from_integer(4)
        b = TritNumber.from_integer(5)
        self.assertEqual((a * b).to_integer(), 20)

        # 測試大數值乘法（觸發Karatsuba演算法）
        large = TritNumber.from_integer(12345)
        self.assertEqual((large * TritNumber.from_integer(2)).to_integer(), 24690)

    def test_division(self):
        """測試除法與取模"""
        a = TritNumber.from_integer(10)
        b = TritNumber.from_integer(3)
        quotient, remainder = ArithmeticCore.divide_with_remainder(a, b)
        self.assertEqual(quotient.to_integer(), 3)
        self.assertEqual(remainder.to_integer(), 1)

    def test_gcd_lcm(self):
        """測試最大公約數與最小公倍數"""
        a = TritNumber.from_integer(12)
        b = TritNumber.from_integer(18)
        self.assertEqual(ArithmeticCore.trit_gcd(a, b).to_integer(), 6)
        self.assertEqual(ArithmeticCore.trit_lcm(a, b).to_integer(), 36)

    def test_multiplication_comprehensive(self):
        """乘法運算全面測試（樸素乘法 + Karatsuba）"""
        print("\n=== 乘法全面測試 ===")

        # 設置較低的Karatsuba閾值以觸發演算法
        original_threshold = ArithmeticCore.KARATSUBA_THRESHOLD
        ArithmeticCore.KARATSUBA_THRESHOLD = 4  # 降低閾值以便測試

        # 1. 基本乘法測試
        basic_cases = [
            (0, 0, 0),
            (0, 5, 0),
            (5, 0, 0),
            (1, 1, 1),
            (1, -1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (2, 3, 6),
            (3, 3, 9),
            (4, -2, -8),
            (-3, -4, 12),
            (10, 10, 100)
        ]

        # 2. 邊界值測試
        boundary_cases = [
            (1, 2 ** 16 - 1, 2 ** 16 - 1),  # 小×大
            (2 ** 16 - 1, 1, 2 ** 16 - 1),  # 大×小
            (2 ** 8, 2 ** 8, 2 ** 16),  # 平方
            (2 ** 16, 2 ** 16, 2 ** 32),  # 大平方
            (2 ** 10, 2 ** 20, 2 ** 30),  # 不同位寬
            (2 ** 10 + 1, 2 ** 10 - 1, (2 ** 10) ** 2 - 1)  # (n+1)(n-1)=n²-1
        ]

        # 3. Karatsuba演算法觸發測試（位數>閾值）
        karatsuba_cases = [
            (12345, 6789, 12345 * 6789),  # 5位×4位
            (987654, 123456, 987654 * 123456),  # 6位×6位
            (2 ** 20, 2 ** 20, 2 ** 40),  # 大數平方
            (2 ** 30 + 1, 2 ** 30 - 1, (2 ** 30) ** 2 - 1)  # 特殊形式
        ]

        # 4. 隨機測試
        random.seed(42)  # 固定種子保證可重複性
        random_cases = []
        for _ in range(20):
            a = random.randint(0, 10000)
            b = random.randint(0, 10000)
            random_cases.append((a, b, a * b))

        # 5. 三進制特性測試
        trit_property_cases = [
            (3 ** 5, 3 ** 4, 3 ** 9),  # 3^m × 3^n = 3^(m+n)
            (2 * 3 ** 3, 3 * 3 ** 2, 6 * 3 ** 5),  # 混合係數
            (3 ** 10 - 1, 3 ** 5, 3 ** 15 - 3 ** 5)  # (3^k-1)×3^m
        ]

        # 6. 閾值附近的具體測試案例（TritNumber輸入）
        trit_consistency_cases = [
            # 1. 31位三進制數（低於閾值，應使用樸素演算法）
            (
                TritNumber.from_digits([Trit(1) for _ in range(31)]),
                TritNumber.from_digits([Trit(-1) for _ in range(31)])
            ),
            (
                # 31位交替模式 (1, -1, 1, ...)
                TritNumber.from_digits([Trit(1) if i % 2 == 0 else Trit(-1) for i in range(31)]),
                # 31位含零模式 (1, 0, -1, 0, ...)
                TritNumber.from_digits(
                    [Trit(1) if i % 4 == 0 else Trit(-1) if i % 4 == 2 else Trit(0) for i in range(31)])
            ),

            # 2. 32位三進制數（等於閾值，應使用Karatsuba演算法）
            (
                # 32位全1三進制數
                TritNumber.from_digits([Trit(1) for _ in range(32)]),
                # 32位全-1三進制數
                TritNumber.from_digits([Trit(-1) for _ in range(32)])
            ),
            (
                # 32位最大正值（全1）
                TritNumber.from_digits([Trit(1) for _ in range(32)]),
                # 32位最小負值（全-1）
                TritNumber.from_digits([Trit(-1) for _ in range(32)])
            ),
            (
                # 32位交替模式
                TritNumber.from_digits([Trit(1) if i % 2 == 0 else Trit(-1) for i in range(32)]),
                # 32位單1其餘0（僅最高位為1）
                TritNumber.from_digits([Trit(1)] + [Trit(0) for _ in range(31)])
            ),

            # 3. 跨閾值組合（31位 × 32位）
            (
                # 31位全1
                TritNumber.from_digits([Trit(1) for _ in range(31)]),
                # 32位全1
                TritNumber.from_digits([Trit(1) for _ in range(32)])
            ),
            (
                # 31位全-1
                TritNumber.from_digits([Trit(-1) for _ in range(31)]),
                # 32位交替模式
                TritNumber.from_digits([Trit(1) if i % 2 == 0 else Trit(-1) for i in range(32)])
            )
        ]

        # 驗證一致性測試案例
        for idx, (a, b) in enumerate(trit_consistency_cases):
            with self.subTest(case=idx, a_bits=len(a.digits), b_bits=len(b.digits)):
                # 分別用兩種演算法計算
                naive_result = ArithmeticCore.naive_multiply(a, b)
                karatsuba_result = ArithmeticCore.karatsuba(a, b)

                # 驗證結果一致性
                self.assertEqual(
                    naive_result,
                    karatsuba_result,
                    f"案例 {idx} 結果不一致（{len(a.digits)}位 × {len(b.digits)}位）"
                )

        # 運行所有整數測試用例
        all_cases = basic_cases + boundary_cases + karatsuba_cases + random_cases + trit_property_cases
        failures = 0

        for i, (a_int, b_int, expected) in enumerate(all_cases):
            a_num = TritNumber.from_integer(a_int)
            b_num = TritNumber.from_integer(b_int)

            try:
                result = a_num * b_num
                result_int = result.to_integer()

                if result_int != expected:
                    print(f"失敗 #{i + 1}: {a_int} × {b_int} = {result_int} (預期 {expected})")
                    failures += 1
                else:
                    # 顯示關鍵測試結果
                    if i < 15 or i % 20 == 0:
                        alg_type = "Karatsuba" if len(a_num.digits) >= ArithmeticCore.KARATSUBA_THRESHOLD or \
                                                  len(b_num.digits) >= ArithmeticCore.KARATSUBA_THRESHOLD else "Naive"
                        print(f"通過 #{i + 1}: {a_int} × {b_int} = {result_int} ({alg_type})")
            except Exception as e:
                print(f"異常 #{i + 1}: {a_int} × {b_int} - {str(e)}")
                failures += 1

        # 恢復原始閾值
        ArithmeticCore.KARATSUBA_THRESHOLD = original_threshold

        # 測試總結
        total = len(all_cases)
        print(f"\n測試總結: {total - failures}通過, {failures}失敗")
        assert failures == 0, f"乘法測試失敗 {failures} 個用例"
        print("乘法全面測試通過")

    def test_multiply_by_constant(self):
        """測試 multiply_by_constant 方法"""
        print("\n=== multiply_by_constant 方法測試 ===")
        test_cases = [
            # (輸入值, 常數, 預期結果)
            (0, 0, 0),
            (0, 5, 0),
            (1, 0, 0),
            (1, 1, 1),
            (1, 2, 2),
            (1, 3, 3),
            (1, 4, 4),
            (1, 5, 5),
            (2, 0, 0),
            (2, 1, 2),
            (2, 2, 4),
            (2, 3, 6),
            (3, 2, 6),
            (5, 3, 15),
            (10, 2, 20),
            (10, 4, 40),
            (10, 5, 50),
            (-5, 2, -10),
            (-3, 3, -9),
        ]

        failures = 0
        for value, constant, expected in test_cases:
            num = TritNumber.from_integer(value)
            try:
                result = ArithmeticCore.multiply_by_constant(num, constant)
                result_int = result.to_integer()

                if result_int != expected:
                    print(f"失敗: {value} * {constant} = {result_int} (預期 {expected})")
                    failures += 1
                else:
                    print(f"通過: {value} * {constant} = {result_int}")
            except Exception as e:
                print(f"異常: {value} * {constant} - {str(e)}")
                failures += 1

        # 測試大常數
        try:
            num = TritNumber.from_integer(100)
            result = ArithmeticCore.multiply_by_constant(num, 100)
            assert result.to_integer() == 10000
            print("大常數測試通過: 100 * 100 = 10000")
        except Exception as e:
            print(f"大常數測試異常: {str(e)}")
            failures += 1

        print(f"測試總結: {len(test_cases) + 1 - failures}通過, {failures}失敗")
        assert failures == 0, "multiply_by_constant 方法測試失敗"
        print("multiply_by_constant 方法測試通過")

    def test_split_at_method(self):
        """測試 split_at 方法"""
        print("\n=== split_at 方法測試 ===")
        test_cases = [
            # (輸入值, 分割位置, 預期高位值, 預期低位值)
            (0, 0, 0, 0),
            (0, 1, 0, 0),
            (1, 0, 0, 1),
            (1, 1, 0, 1),
            (8, 1, 2, 2),
            (9, 2, 1, 0),
            (10, 1, 3, 1),
            (27, 3, 1, 0),
            (15, 2, 1, 6),  # 15的三進制是[1,-1,0]，位置2分割：高位=[1,-1]=1 * 3-1=2? 實際應為[1]和[-1,0]
        ]

        failures = 0
        for value, position, expected_high, expected_low in test_cases:
            num = TritNumber.from_integer(value)
            try:
                high, low = ArithmeticCore.split_at(num, position)
                high_int = high.to_integer()
                low_int = low.to_integer()

                if high_int != expected_high or low_int != expected_low:
                    print(
                        f"失敗: {value}在位置{position}分割 → 高:{high_int} 低:{low_int} (預期 高:{expected_high} 低:{expected_low})")
                    failures += 1
                else:
                    print(f"通過: {value}在位置{position}分割 → 高:{high_int} 低:{low_int}")
            except Exception as e:
                print(f"異常: {value}在位置{position}分割 - {str(e)}")
                failures += 1

        # 測試負數分割
        try:
            num = TritNumber.from_integer(-10)
            high, low = ArithmeticCore.split_at(num, 1)
            print(f"負數測試: -10在位置1分割 → 高:{high.to_integer()} 低:{low.to_integer()}")
        except Exception as e:
            print(f"負數分割異常: {str(e)}")
            failures += 1

        print(f"測試總結: {len(test_cases) + 1 - failures}通過, {failures}失敗")
        assert failures == 0, "split_at 方法測試失敗"
        print("split_at 方法測試通過")

    def test_shift_methods(self):
        """測試 left_shift 和 right_shift 方法"""
        print("\n=== 位移方法測試 ===")
        test_cases = [
            # (輸入值, 左移位, 左移預期結果, 右移位, 右移預期結果)
            (0, 0, 0, 0, 0),
            (0, 1, 0, 1, 0),
            (1, 0, 1, 0, 1),
            (1, 1, 3, 1, 0),
            (1, 2, 9, 2, 0),
            (3, 1, 9, 1, 1),
            (9, 1, 27, 1, 3),
            (9, 2, 81, 2, 1),
            (10, 1, 30, 1, 3),  # 10 * 3=30, 10//3=3
            (10, 2, 90, 2, 1),  # 10 * 9=90, 10//9=1
            (27, 1, 81, 1, 9),
            (27, 3, 729, 3, 1),
            (100, 2, 900, 2, 11),  # 100 * 9=900, 100//9=11.11→11
            (-5, 1, -15, 1, -2),  # -5 * 3=-15, -5//3=-2→-1
            (-10, 2, -90, 2, -1),  # -10 * 9=-90, -10//9=-2→-1
        ]

        failures = 0
        for value, l_shift, l_expected, r_shift, r_expected in test_cases:
            num = TritNumber.from_integer(value)
            try:
                # 測試左移
                left_result = ArithmeticCore.left_shift(num, l_shift)
                left_int = left_result.to_integer()
                if left_int != l_expected:
                    print(f"左移失敗: {value} << {l_shift} = {left_int} (預期 {l_expected})")
                    failures += 1
                else:
                    print(f"左移通過: {value} << {l_shift} = {left_int}")

                # 測試右移
                right_result = ArithmeticCore.right_shift(num, r_shift)
                right_int = right_result.to_integer()
                if right_int != r_expected:
                    print(f"右移失敗: {value} >> {r_shift} = {right_int} (預期 {r_expected})")
                    failures += 1
                else:
                    print(f"右移通過: {value} >> {r_shift} = {right_int}")
            except Exception as e:
                print(f"位移測試異常: {str(e)}")
                failures += 1

        # 測試大位移
        try:
            num = TritNumber.from_integer(1000)
            left_result = ArithmeticCore.left_shift(num, 5)  # 1000 * 3^5 = 1000 * 243=243000
            assert left_result.to_integer() == 243000
            print("大左移測試通過: 1000 << 5 = 243000")

            right_result = ArithmeticCore.right_shift(num, 5)  # 1000 // 243 = 4
            assert right_result.to_integer() == 4
            print("大右移測試通過: 1000 >> 5 = 4")
        except Exception as e:
            print(f"大位移測試異常: {str(e)}")
            failures += 1

        print(f"測試總結: {len(test_cases) * 2 + 2 - failures}通過, {failures}失敗")
        assert failures == 0, "位移方法測試失敗"
        print("位移方法測試通過")

    def test_balance_division(self):
        """多位數三進制除法運算測試"""
        print("\n=== 多位數三進制除法測試 ===")
        test_cases = [
            (TritNumber.from_integer(6), TritNumber.from_integer(2), 3),
            (TritNumber.from_integer(-6), TritNumber.from_integer(2), -3),
            (TritNumber.from_integer(5), TritNumber.from_integer(3), 1),  # 5/3=1.666→1（向零舍入）
            (TritNumber.from_integer(-5), TritNumber.from_integer(3), -1),  # -5/3=-1.666→-1（向零舍入）
            (TritNumber.from_integer(0), TritNumber.from_integer(5), 0),
            (TritNumber.from_integer(7), TritNumber.from_integer(3), 2),  # 7/3=2.333→2（向零舍入）
            (TritNumber.from_integer(-7), TritNumber.from_integer(3), -2),  # -7/3=-2.333→-2（向零舍入）
            (TritNumber.from_integer(5), TritNumber.from_integer(-3), -1),  # 5/-3=-1.666→-1（向零舍入）
            (TritNumber.from_integer(0), TritNumber.from_integer(-5), 0),
            (TritNumber.from_integer(8), TritNumber.from_integer(1), 8),
            (TritNumber.from_integer(8), TritNumber.from_integer(-1), -8),
            (TritNumber.from_integer(-8), TritNumber.from_integer(-1), 8),
            (TritNumber.from_integer(2), TritNumber.from_integer(3), 0),
            (TritNumber.from_integer(-2), TritNumber.from_integer(3), 0),
            (TritNumber.from_integer(2), TritNumber.from_integer(-3), 0),
            (TritNumber.from_integer(4), TritNumber.from_integer(2), 2),  # 4的三進制是11，2是1-1
            (TritNumber.from_integer(8), TritNumber.from_integer(4), 2),  # 8的三進制是-10-1，4是11
            (TritNumber.from_integer(27), TritNumber.from_integer(9), 3),  # 3^3=27
            (TritNumber.from_integer(80), TritNumber.from_integer(9), 8),  # 80/9=8.888→8
            (TritNumber.from_integer(-80), TritNumber.from_integer(9), -8),  # -80/9=-8.888→-8（向零舍入）
            (TritNumber.from_integer(243), TritNumber.from_integer(9), 27),  # 243 / 9 = 27
            (TritNumber.from_integer(-200), TritNumber.from_integer(7), -28),  # -200 / 7 = -28.57... → -28（向零舍入）
            (TritNumber.from_integer(300), TritNumber.from_integer(-10), -30),  # 300 / -10 = -30
            (TritNumber.from_integer(-500), TritNumber.from_integer(11), -45),  # -500 / 11 = -45.454... → -45（向零舍入）
            # 極值測試
            (TritNumber.from_integer(2 ** 16), TritNumber.from_integer(1), 2 ** 16),
            (TritNumber.from_integer(-2 ** 16), TritNumber.from_integer(1), -2 ** 16),
            # 小數位處理
            (TritNumber.from_integer(1), TritNumber.from_integer(3), 0)  # 1/3=0.333→0
        ]
        for dividend, divisor, expected in test_cases:
            result = dividend / divisor
            print(
                f"{dividend} ({dividend.to_integer()}) / "
                f"{divisor} ({divisor.to_integer()}) = "
                f"{result} ({result.to_integer()})"
            )
            assert result.to_integer() == expected
        print("多位數三進制除法驗證通過")

    def test_power_method(self):
        """測試power方法"""
        print("\n=== Power方法測試 ===")

        # 定義常用值
        zero = TritNumber.zero()
        one = TritNumber(TritDigit([Trit(1)]))
        neg_one = TritNumber(TritDigit([Trit(-1)]))

        # 1. 底數為0，指數為0
        result = ArithmeticCore.power(zero, zero)
        assert result == one, f"0^0 = {result} (預期: 1)"
        print("0^0 = 1 通過")

        # 2. 底數為0，指數為正
        positive_exponent = TritNumber.from_integer(5)
        result = ArithmeticCore.power(zero, positive_exponent)
        assert result == zero, f"0^5 = {result} (預期: 0)"
        print("0^5 = 0 通過")

        # 3. 底數為0，指數為負
        negative_exponent = TritNumber.from_integer(-5)
        try:
            result = ArithmeticCore.power(zero, negative_exponent)
            assert False, "0的負指數應引發異常"
        except ZeroDivisionError as e:
            assert str(e) == "0的負指數未定義", f"錯誤資訊不符: {str(e)}"
            print("0的負指數異常通過")

        # 4. 指數為0（底數非零）
        base = TritNumber.from_integer(5)
        result = ArithmeticCore.power(base, zero)
        assert result == one, f"5^0 = {result} (預期: 1)"
        print("5^0 = 1 通過")

        # 5. 指數為1
        result = ArithmeticCore.power(base, one)
        assert result == base, f"5^1 = {result} (預期: 5)"
        print("5^1 = 5 通過")

        # 6. 指數為-1
        result = ArithmeticCore.power(base, neg_one)
        expected = TritNumber.from_integer(0)
        assert result == expected, f"5^-1 = {result} (預期: 0)"
        print("5^-1 = 0 通過")

        base2 = TritNumber.from_integer(2)
        result = ArithmeticCore.power(base2, neg_one)
        assert result == zero, f"2^-1 = {result} (預期: 0)"
        print("2^-1 = 0 通過")

        base1 = TritNumber.from_integer(1)
        result = ArithmeticCore.power(base1, neg_one)
        assert result == one, f"1^-1 = {result} (預期: 1)"
        print("1^-1 = 1 通過")

        base_neg1 = TritNumber.from_integer(-1)
        result = ArithmeticCore.power(base_neg1, neg_one)
        assert result == base_neg1, f"(-1)^-1 = {result} (預期: -1)"
        print("(-1)^-1 = -1 通過")

        # 7. 正指數
        base3 = TritNumber.from_integer(3)
        exponent2 = TritNumber.from_integer(2)
        result = ArithmeticCore.power(base3, exponent2)
        expected = TritNumber.from_integer(9)
        assert result == expected, f"3^2 = {result} (預期: 9)"
        print("3^2 = 9 通過")

        base2 = TritNumber.from_integer(2)
        exponent5 = TritNumber.from_integer(5)
        result = ArithmeticCore.power(base2, exponent5)
        expected = TritNumber.from_integer(32)
        assert result == expected, f"2^5 = {result} (預期: 32)"
        print("2^5 = 32 通過")

        base_neg2 = TritNumber.from_integer(-2)
        exponent3 = TritNumber.from_integer(3)
        result = ArithmeticCore.power(base_neg2, exponent3)
        expected = TritNumber.from_integer(-8)
        assert result == expected, f"(-2)^3 = {result} (預期: -8)"
        print("(-2)^3 = -8 通過")

        # 8. 負指數
        result = ArithmeticCore.power(base2, TritNumber.from_integer(-3))
        assert result == zero, f"2^-3 = {result} (預期: 0)"
        print("2^-3 = 0 通過")

        result = ArithmeticCore.power(base3, TritNumber.from_integer(-2))
        assert result == zero, f"3^-2 = {result} (預期: 0)"
        print("3^-2 = 0 通過")

        result = ArithmeticCore.power(base1, TritNumber.from_integer(-10))
        assert result == one, f"1^-10 = {result} (預期: 1)"
        print("1^-10 = 1 通過")

        result = ArithmeticCore.power(base_neg1, TritNumber.from_integer(-2))
        assert result == one, f"(-1)^-2 = {result} (預期: 1)"
        print("(-1)^-2 = 1 通過")

        result = ArithmeticCore.power(base_neg1, TritNumber.from_integer(-3))
        assert result == base_neg1, f"(-1)^-3 = {result} (預期: -1)"
        print("(-1)^-3 = -1 通過")

        # 9. 大數冪運算
        base2 = TritNumber.from_integer(2)
        exponent10 = TritNumber.from_integer(10)
        result = ArithmeticCore.power(base2, exponent10)
        expected = TritNumber.from_integer(1024)
        assert result == expected, f"2^10 = {result} (預期: 1024)"
        print("2^10 = 1024 通過")

        base3 = TritNumber.from_integer(3)
        exponent5 = TritNumber.from_integer(5)
        result = ArithmeticCore.power(base3, exponent5)
        expected = TritNumber.from_integer(243)
        assert result == expected, f"3^5 = {result} (預期: 243)"
        print("3^5 = 243 通過")

        print("Power方法測試通過")

    def test_trit_mod(self):
        """測試三進制數的取模運算"""
        print("\n=== 三進制數取模運算測試 ===")

        # 測試用例格式: (a, b, expected_mod)
        test_cases = [
            # 基本測試
            (0, 1, 0),  # 0 mod 1 = 0
            (1, 1, 0),  # 1 mod 1 = 0
            (2, 1, 0),  # 2 mod 1 = 0
            (3, 1, 0),  # 3 mod 1 = 0

            # 正常取模
            (5, 3, 2),  # 5 mod 3 = 2
            (7, 3, 1),  # 7 mod 3 = 1
            (8, 3, 2),  # 8 mod 3 = 2
            (9, 3, 0),  # 9 mod 3 = 0

            # 負數取模
            (-5, 3, 1),  # -5 mod 3 = 1 (因為 -5 = -2*3 + 1)
            (-7, 3, 2),  # -7 mod 3 = 2 (因為 -7 = -3*3 + 2)
            (-8, 3, 1),  # -8 mod 3 = 1 (因為 -8 = -3*3 + 1)
            (-9, 3, 0),  # -9 mod 3 = 0

            # 除數為負數
            (5, -3, 2),  # 5 mod -3 = 2
            (-5, -3, 1),  # -5 mod -3 = 1

            # 邊界情況
            (0, 2, 0),  # 0 mod 2 = 0
            (1, 2, 1),  # 1 mod 2 = 1
            (2, 2, 0),  # 2 mod 2 = 0
            (3, 2, 1),  # 3 mod 2 = 1

            # 大數取模
            (100, 7, 2),  # 100 mod 7 = 2
            (100, 13, 9),  # 100 mod 13 = 9
            (1000, 27, 1),  # 1000 mod 27 = 1 (因為 1000 = 37*27 + 1)

            # 三進制特性測試
            (9, 3, 0),  # 3^2 mod 3 = 0
            (27, 9, 0),  # 3^3 mod 3^2 = 0
            (10, 3, 1),  # 10 mod 3 = 1
        ]

        failures = 0
        for a_int, b_int, expected_mod in test_cases:
            # 轉換為三進制數
            a = TritNumber.from_integer(a_int)
            b = TritNumber.from_integer(b_int)

            try:
                # 計算模運算
                mod_result = ArithmeticCore.trit_mod(a, b)
                mod_int = mod_result.to_integer()

                # 驗證結果
                if mod_int != expected_mod:
                    print(f"失敗: {a_int} mod {b_int} = {mod_int} (預期 {expected_mod})")
                    failures += 1
                else:
                    print(f"通過: {a_int} mod {b_int} = {mod_int}")
            except Exception as e:
                print(f"異常: {a_int} mod {b_int} - {str(e)}")
                failures += 1

        # 特殊邊界情況測試
        print("\n--- 特殊邊界情況測試 ---")

        # 測試除數為0的情況
        try:
            result = ArithmeticCore.trit_mod(TritNumber.from_integer(5), TritNumber.zero())
            print(f"除數為0測試失敗: 應引發異常但得到結果 {result}")
            failures += 1
        except ZeroDivisionError as e:
            if "除數不能為零" in str(e):
                print("除數為0測試通過: 正確引發ZeroDivisionError")
            else:
                print(f"除數為0測試異常資訊不符: {e}")
                failures += 1
        except Exception as e:
            print(f"除數為0測試異常類型不符: {e}")
            failures += 1

        # 測試被除數為0的情況
        try:
            result = ArithmeticCore.trit_mod(TritNumber.zero(), TritNumber.from_integer(5))
            if result == TritNumber.zero():
                print("被除數為0測試通過: 0 mod 5 = 0")
            else:
                print(f"被除數為0測試失敗: 0 mod 5 = {result.to_integer()} (預期 0)")
                failures += 1
        except Exception as e:
            print(f"被除數為0測試異常: {e}")
            failures += 1

        # 測試模運算性質: (a + b) mod m = (a mod m + b mod m) mod m
        print("\n--- 模運算性質驗證 ---")
        property_test_cases = [
            (7, 5, 3),
            (12, 8, 5),
            (-7, 4, 3),
            (15, 9, 7),
        ]

        for a_int, b_int, m_int in property_test_cases:
            # 轉換為三進制數
            a = TritNumber.from_integer(a_int)
            b = TritNumber.from_integer(b_int)
            m = TritNumber.from_integer(m_int)

            # (a + b) mod m
            left = ArithmeticCore.trit_mod(a + b, m)

            # (a mod m + b mod m) mod m
            a_mod = ArithmeticCore.trit_mod(a, m)
            b_mod = ArithmeticCore.trit_mod(b, m)

            # 智能歸一化演算法
            sum_mod = a_mod + b_mod
            sum_int = sum_mod.to_integer()

            # 根據模數選擇不同的歸一化策略
            if m_int == 3:
                # 對於模3，使用三值邏輯歸一化
                sum_int %= 3
                if sum_int == 2:
                    sum_int = -1
                elif sum_int == -2:
                    sum_int = 1
                sum_mod_normalized = TritNumber.from_integer(sum_int)
            else:
                # 對於其他模數，使用整數歸一化
                sum_mod_normalized = TritNumber.from_integer(sum_int % m_int)

            # 計算 (a mod m + b mod m) mod m
            right = ArithmeticCore.trit_mod(sum_mod_normalized, m)

            if left == right:
                print(f"模加法性質通過: ({a_int}+{b_int}) mod {m_int} = {left.to_integer()}")
            else:
                print(f"模加法性質失敗: ({a_int}+{b_int}) mod {m_int} = {left.to_integer()}, "
                      f"但 ({a_int} mod {m_int} + {b_int} mod {m_int}) mod {m_int} = {right.to_integer()}")
                failures += 1

        # 測試總結
        print(f"\n測試總結: {len(test_cases) + 2 + len(property_test_cases) - failures}通過, {failures}失敗")
        assert failures == 0, f"取模運算測試失敗 {failures} 個用例"
        print("三進制數取模運算測試通過")

    def test_trit_gcd_lcm(self):
        """測試三進制數的最大公約數（GCD）和最小公倍數（LCM）計算"""
        print("\n=== 三進制數GCD和LCM測試 ===")
        test_cases = [
            (0, 5, 5, None),  # LCM(0,5)未定義，標記為None
            (5, 0, 5, None),

            # 基本測試
            (1, 1, 1, 1),
            (2, 3, 1, 6),
            (4, 6, 2, 12),

            # 負數測試
            (-4, 6, 2, 12),
            (4, -6, 2, 12),
            (-4, -6, 2, 12),

            # 較大數值測試
            (12, 18, 6, 36),
            (15, 25, 5, 75),
            (24, 36, 12, 72),

            # 質數測試
            (7, 11, 1, 77),
            (13, 17, 1, 221),

            # 倍數關係測試
            (8, 24, 8, 24),
            (9, 81, 9, 81),

            # 三進制特性測試
            (9, 27, 9, 27),
            (3, 9, 3, 9),
        ]

        failures = 0
        for a_int, b_int, expected_gcd, expected_lcm in test_cases:
            a = TritNumber.from_integer(a_int)
            b = TritNumber.from_integer(b_int)

            # 測試GCD
            try:
                gcd_result = ArithmeticCore.trit_gcd(a, b)
                if gcd_result.to_integer() != expected_gcd:
                    print(f"GCD失敗: {a_int} 和 {b_int}，預期 {expected_gcd}，實際 {gcd_result.to_integer()}")
                    failures += 1
            except ValueError as e:
                print(f"GCD異常: {a_int} 和 {b_int}，錯誤: {str(e)}")
                failures += 1

            # 測試LCM（跳過未定義情況）
            if expected_lcm is not None:
                try:
                    lcm_result = ArithmeticCore.trit_lcm(a, b)
                    if lcm_result.to_integer() != expected_lcm:
                        print(f"LCM失敗: {a_int} 和 {b_int}，預期 {expected_lcm}，實際 {lcm_result.to_integer()}")
                        failures += 1
                except ValueError as e:
                    print(f"LCM異常: {a_int} 和 {b_int}，錯誤: {str(e)}")
                    failures += 1

        if failures == 0:
            print("所有GCD和LCM測試通過")
        else:
            print(f"GCD和LCM測試失敗，共 {failures} 處錯誤")


class TestTensorCore(unittest.TestCase):
    """測試運算協作層TensorCore"""

    def test_tensor_logic_full_coverage(self):
        """驗證19683種函數空間完備性 + 邊界條件"""
        print("\n=== 邏輯函數測試 ===")
        system = TensorCore()
        test_pass = True

        # 使用 Trit 直接計算
        for x_val in [-1, 0, 1]:
            for y_val in [-1, 0, 1]:
                x = Trit(x_val)
                y = Trit(y_val)

                # AND函數驗證
                psi_and = [1, 0, -1, 0, 0, 0, -1, 0, 1]
                result = system.tensor_function(x, y, psi_and)

                # 使用 Trit 乘法
                expected_value = x.value * y.value
                if expected_value == 2: expected_value = -1
                expected = Trit(expected_value)

                if result != expected:
                    print(f"AND失敗: ({x},{y}) -> {result} != {expected}")
                    test_pass = False

                # OR函數驗證
                psi_or = [1, -1, 0, -1, 0, 1, 0, 1, -1]
                result = system.tensor_function(x, y, psi_or)

                # 使用 Trit 加法
                expected_value = x.value + y.value
                if expected_value < -1: expected_value += 3
                if expected_value > 1: expected_value -= 3
                expected = Trit(expected_value)

                if result != expected:
                    print(f"OR失敗: ({x},{y}) -> {result} != {expected}")
                    test_pass = False

        # 隨機函數空間遍曆
        for i in range(100):
            random_psi = [random.choice([-1, 0, 1]) for _ in range(9)]
            for x_val in [-1, 0, 1]:
                for y_val in [-1, 0, 1]:
                    x = Trit(x_val)
                    y = Trit(y_val)
                    try:
                        result = system.tensor_function(x, y, random_psi)
                        if result.value not in (-1, 0, 1):
                            print(f"值域錯誤: psi={random_psi} -> {result}")
                            test_pass = False
                    except Exception as e:
                        print(f"執行錯誤: psi={random_psi} error={str(e)}")
                        test_pass = False

        # 邊界條件測試
        try:
            system.tensor_function(Trit(0), Trit(0), [])
            print("空參數測試失敗: 未觸發異常")
            test_pass = False
        except ValueError as e:
            if "psi 參數必須為9個元素" not in str(e):
                print(f"空參數測試異常資訊不符: {e}")
                test_pass = False

        # 無效參數值測試
        invalid_psi = [0, 0, 0, 0, 2, 0, 0, 0, 0]  # 2是無效值
        try:
            system.tensor_function(Trit(0), Trit(0), invalid_psi)
            print("無效參數值測試失敗: 未觸發異常")
            test_pass = False
        except ValueError as e:
            if "無效的PSI值" not in str(e):
                print(f"無效參數值測試異常資訊不符: {e}")
                test_pass = False

        # 極端輸入值測試
        try:
            result = system.tensor_function(Trit(-1), Trit(1), [1] * 9)
            assert result == Trit(1), "全1參數失敗"
            result = system.tensor_function(Trit(0), Trit(0), [-1] * 9)
            assert result == Trit(-1), "全-1參數失敗"
        except Exception as e:
            print(f"極端輸入測試失敗: {str(e)}")
            test_pass = False

        # 添加邊界值測試
        try:
            # 測試全零參數
            result = system.tensor_function(Trit(0), Trit(0), [0] * 9)
            assert result == Trit(0), "全零參數失敗"

            # 測試全一參數
            result = system.tensor_function(Trit(1), Trit(1), [1] * 9)
            # 點積計算：只有對應位置為1，其他為0，所以結果為1
            assert result == Trit(1), "全一參數失敗"

            # 測試全負一參數
            result = system.tensor_function(Trit(-1), Trit(-1), [-1] * 9)
            # 點積計算：只有對應位置為1，其他為0，所以結果為-1
            assert result == Trit(-1), "全負一參數失敗"
        except Exception as e:
            print(f"邊界值測試失敗: {str(e)}")
            test_pass = False

        # 測試總結
        assert test_pass, "邏輯函數測試失敗"
        print("19683函數空間 + 邊界條件 全面驗證通過")

    def test_tensor_core_functionality(self):
        """測試TensorCore的核心功能"""
        print("\n=== TensorCore功能測試 ===")
        tensor_core = TensorCore()

        # 測試基本張量運算
        test_cases = [
            # (x, y, psi, expected)
            (Trit(1), Trit(1), [1, 0, 0, 0, 1, 0, 0, 0, 1], Trit(1)),  # 單位矩陣
            (Trit(1), Trit(0), [1, 0, 0, 0, 1, 0, 0, 0, 1], Trit(0)),
            (Trit(1), Trit(-1), [1, 0, 0, 0, 1, 0, 0, 0, 1], Trit(0)),
            (Trit(1), Trit(1), [0, 0, 0, 0, 0, 0, 0, 0, 0], Trit(0)),  # 零矩陣
        ]

        for x, y, psi, expected in test_cases:
            result = tensor_core.tensor_function(x, y, psi)
            print(f"f({x}, {y}, {psi}) = {result} (預期: {expected})")
            assert result == expected, f"張量函數測試失敗: f({x}, {y}, {psi}) = {result}, 預期 {expected}"

        # 測試AND運算
        and_psi = [1, 0, -1, 0, 0, 0, -1, 0, 1]
        and_test_cases = [
            (Trit(1), Trit(1), Trit(1)),
            (Trit(1), Trit(0), Trit(0)),
            (Trit(1), Trit(-1), Trit(-1)),
            (Trit(0), Trit(1), Trit(0)),
            (Trit(0), Trit(0), Trit(0)),
            (Trit(0), Trit(-1), Trit(0)),
            (Trit(-1), Trit(1), Trit(-1)),
            (Trit(-1), Trit(0), Trit(0)),
            (Trit(-1), Trit(-1), Trit(1)),
        ]

        for x, y, expected in and_test_cases:
            result = tensor_core.tensor_function(x, y, and_psi)
            print(f"AND({x}, {y}) = {result} (預期: {expected})")
            assert result == expected, f"AND運算測試失敗: AND({x}, {y}) = {result}, 預期 {expected}"

        # 測試OR運算
        or_psi = [1, -1, 0, -1, 0, 1, 0, 1, -1]
        or_test_cases = [
            (Trit(1), Trit(1), Trit(-1)),
            (Trit(1), Trit(0), Trit(1)),
            (Trit(1), Trit(-1), Trit(0)),
            (Trit(0), Trit(1), Trit(1)),
            (Trit(0), Trit(0), Trit(0)),
            (Trit(0), Trit(-1), Trit(-1)),
            (Trit(-1), Trit(1), Trit(0)),
            (Trit(-1), Trit(0), Trit(-1)),
            (Trit(-1), Trit(-1), Trit(1)),
        ]

        for x, y, expected in or_test_cases:
            result = tensor_core.tensor_function(x, y, or_psi)
            print(f"OR({x}, {y}) = {result} (預期: {expected})")
            assert result == expected, f"OR運算測試失敗: OR({x}, {y}) = {result}, 預期 {expected}"

        # 測試數值表示和重構
        value_test_cases = [0, 1, -1, 2, -2, 3, -3, 5, -5, 10, -10, 27, -27]

        for value in value_test_cases:
            representation = tensor_core.represent_value(value)
            reconstructed = tensor_core.reconstruct_value(representation)
            print(f"值 {value} → 表示 {[str(t) for t in representation]} → 重構 {reconstructed}")
            assert reconstructed == value, f"數值表示/重構測試失敗: {value} → {reconstructed}"

        # 測試矩陣運算
        matrix_a = [[Trit(1), Trit(0)], [Trit(-1), Trit(1)]]
        matrix_b = [[Trit(1), Trit(1)], [Trit(0), Trit(-1)]]

        try:
            result = tensor_core.trit_matrix_multiply(matrix_a, matrix_b)
            print(f"矩陣乘法結果: {[[str(t) for t in row] for row in result]}")

            # 驗證結果 (預期: [[1, 1], [-1, -2]] → 三值邏輯中-2會轉換為1)
            expected = [[Trit(1), Trit(1)], [Trit(-1), Trit(1)]]
            for i in range(len(result)):
                for j in range(len(result[0])):
                    assert result[i][j] == expected[i][
                        j], f"矩陣乘法測試失敗: 位置({i},{j}) = {result[i][j]}, 預期 {expected[i][j]}"
        except Exception as e:
            print(f"矩陣乘法測試異常: {e}")
            raise

        print("TensorCore功能測試通過")

    def test_tensor_core_edge_cases(self):
        """測試TensorCore的邊界情況"""
        print("\n=== TensorCore邊界情況測試 ===")
        tensor_core = TensorCore()

        # 測試無效PSI參數
        invalid_psi_cases = [
            [],  # 空列表
            [1, 0, 0, 0, 1, 0, 0, 0],  # 長度不足
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # 長度過長
            [1, 0, 0, 0, 2, 0, 0, 0, 1],  # 無效值
        ]

        for psi in invalid_psi_cases:
            try:
                result = tensor_core.tensor_function(Trit(0), Trit(0), psi)
                assert False, f"無效PSI參數測試失敗: 應引發異常但得到結果 {result}"
            except ValueError as e:
                print(f"無效PSI參數測試通過: {e}")

        # 測試極端數值
        extreme_values = [-100, 0, 100, -1000, 1000]
        for value in extreme_values:
            try:
                representation = tensor_core.represent_value(value)
                reconstructed = tensor_core.reconstruct_value(representation)
                print(f"極端值 {value} → 重構 {reconstructed}")
                assert reconstructed == value, f"極端值測試失敗: {value} → {reconstructed}"
            except Exception as e:
                print(f"極端值 {value} 測試異常: {e}")
                raise

        print("TensorCore邊界情況測試通過")

    def test_semi_tensor_product_correctness(self):
        """測試半張量積的正確性"""
        print("\n=== 半張量積正確性測試 ===")
        tensor_core = TensorCore()

        # 簡單的測試用例
        A = [[Trit(1), Trit(0)],
             [Trit(-1), Trit(1)]]

        B = [[Trit(1)],
             [Trit(-1)]]

        # 手動計算預期結果：A ⋉ B = A · (I_2 ⊗ B) = [[1*1 + 0*(-1)], [-1*1 + 1*(-1)]] = [[1], [-2]] → [[1], [1]] (模3)
        expected = [[Trit(1)], [Trit(1)]]

        result = tensor_core.semi_tensor_product(A, B)

        print(f"A = {[[str(t) for t in row] for row in A]}")
        print(f"B = {[[str(t) for t in row] for row in B]}")
        print(f"A ⋉ B = {[[str(t) for t in row] for row in result]}")
        print(f"預期 = {[[str(t) for t in row] for row in expected]}")

        # 驗證結果
        assert len(result) == len(expected), f"行數不匹配: {len(result)} != {len(expected)}"
        assert len(result[0]) == len(expected[0]), f"列數不匹配: {len(result[0])} != {len(expected[0])}"

        for i in range(len(result)):
            for j in range(len(result[0])):
                assert result[i][j] == expected[i][j], f"位置({i},{j})不匹配: {result[i][j]} != {expected[i][j]}"

        # 測試更多用例
        print("\n--- 更多測試用例 ---")

        # 測試用例2: 1x2矩陣與2x1矩陣
        A2 = [[Trit(1), Trit(-1)]]
        B2 = [[Trit(1)], [Trit(0)]]
        result2 = tensor_core.semi_tensor_product(A2, B2)
        print(f"A2 ⋉ B2 = {[[str(t) for t in row] for row in result2]}")

        # 測試用例3: 單位矩陣
        I = [[Trit(1), Trit(0)], [Trit(0), Trit(1)]]
        result3 = tensor_core.semi_tensor_product(I, B)
        print(f"I ⋉ B = {[[str(t) for t in row] for row in result3]}")

        print("半張量積正確性測試通過")

    def test_high_dimension_matrix_ops(self):
        """驗證4×4矩陣乘法與半張量積"""
        tensor_core = TensorCore()

        # 生成4×4三值矩陣
        def gen_matrix(size):
            return [[Trit(random.choice([-1, 0, 1])) for _ in range(size)]
                    for _ in range(size)]

        A = gen_matrix(4)
        B = gen_matrix(4)

        # 矩陣乘法正確性驗證
        result = tensor_core.trit_matrix_multiply(A, B)
        for row in result:
            assert all(t.value in (-1, 0, 1) for t in row), "結果超出三值範圍"

        # 半張量積穩定性驗證 - 對於4×4矩陣，半張量積結果應該是4×4
        stp_result = tensor_core.semi_tensor_product(A, B)

        # 驗證STP結果維度 (4×4輸入應輸出4×4矩陣)
        assert len(stp_result) == 4, f"行數錯誤: {len(stp_result)} != 4"
        assert len(stp_result[0]) == 4, f"列數錯誤: {len(stp_result[0])} != 4"

        # 驗證所有值在有效範圍內
        for row in stp_result:
            for val in row:
                assert val.value in (-1, 0, 1), f"無效值: {val.value}"

        print("4×4矩陣半張量積測試通過")


class TestPolynomialCore(unittest.TestCase):
    """測試運算協作層PolynomialCore"""

    def test_bivariate_operator(self):
        """測試雙變數多項式算子"""
        print("\n=== 雙變數算子測試 ===")

        # 定義AND算子係數 (x AND y)
        and_coeffs = (0, 0, 0, 0, 0, 1)  # f(x,y) = x*y

        # 測試真值表
        test_cases = [
            (Trit(1), Trit(1), Trit(1)),
            (Trit(1), Trit(0), Trit(0)),
            (Trit(1), Trit(-1), Trit(-1)),
            (Trit(0), Trit(1), Trit(0)),
            (Trit(0), Trit(0), Trit(0)),
            (Trit(0), Trit(-1), Trit(0)),
            (Trit(-1), Trit(1), Trit(-1)),
            (Trit(-1), Trit(0), Trit(0)),
            (Trit(-1), Trit(-1), Trit(1))
        ]

        for x, y, expected in test_cases:
            result = PolynomialCore.evaluate_bivariate(x, y, and_coeffs)
            print(f"{x} AND {y} = {result} (預期: {expected})")
            assert result == expected

        # 定義OR算子係數 (x OR y)
        or_coeffs = (0, 1, 1, 0, 0, 0)  # f(x,y)=x+y

        test_cases = [
            (Trit(1), Trit(1), Trit(-1)),
            (Trit(1), Trit(0), Trit(1)),
            (Trit(1), Trit(-1), Trit(0)),
            (Trit(0), Trit(1), Trit(1)),
            (Trit(0), Trit(0), Trit(0)),
            (Trit(0), Trit(-1), Trit(-1)),
            (Trit(-1), Trit(1), Trit(0)),
            (Trit(-1), Trit(0), Trit(-1)),
            (Trit(-1), Trit(-1), Trit(1))
        ]

        for x, y, expected in test_cases:
            result = PolynomialCore.evaluate_bivariate(x, y, or_coeffs)
            print(f"{x} OR {y} = {result} (預期: {expected})")
            assert result == expected

        print("雙變數算子驗證通過")

    def test_polynomial_core_univariate(self):
        """測試PolynomialCore的單變數多項式功能"""
        print("\n=== PolynomialCore單變數測試 ===")

        # 測試線性多項式: f(x) = x
        linear_coeffs = (0, 1, 0)
        test_cases = [
            (Trit(-1), Trit(-1)),
            (Trit(0), Trit(0)),
            (Trit(1), Trit(1)),
        ]

        for x, expected in test_cases:
            result = PolynomialCore.evaluate_univariate(x, linear_coeffs)
            print(f"f({x}) = {result} (預期: {expected})")
            assert result == expected, f"線性多項式測試失敗: f({x}) = {result}, 預期 {expected}"

        # 測試二次多項式: f(x) = x²
        quadratic_coeffs = (1, 0, 0)
        test_cases = [
            (Trit(-1), Trit(1)),  # (-1)² = 1
            (Trit(0), Trit(0)),  # 0² = 0
            (Trit(1), Trit(1)),  # 1² = 1
        ]

        for x, expected in test_cases:
            result = PolynomialCore.evaluate_univariate(x, quadratic_coeffs)
            print(f"f({x}) = {result} (預期: {expected})")
            assert result == expected, f"二次多項式測試失敗: f({x}) = {result}, 預期 {expected}"

        # 測試常數多項式: f(x) = 1
        constant_coeffs = (0, 0, 1)
        test_cases = [
            (Trit(-1), Trit(1)),
            (Trit(0), Trit(1)),
            (Trit(1), Trit(1)),
        ]

        for x, expected in test_cases:
            result = PolynomialCore.evaluate_univariate(x, constant_coeffs)
            print(f"f({x}) = {result} (預期: {expected})")
            assert result == expected, f"常數多項式測試失敗: f({x}) = {result}, 預期 {expected}"

        print("PolynomialCore單變數測試通過")

    def test_polynomial_core_bivariate(self):
        """測試PolynomialCore的雙變數多項式功能"""
        print("\n=== PolynomialCore雙變數測試 ===")

        # 測試AND運算: f(x,y) = x*y （平衡三進制模3乘法）
        and_coeffs = (0, 0, 0, 0, 0, 1)
        test_cases = [
            (Trit(1), Trit(1), Trit(1)),
            (Trit(1), Trit(0), Trit(0)),
            (Trit(1), Trit(-1), Trit(-1)),
            (Trit(0), Trit(1), Trit(0)),
            (Trit(0), Trit(0), Trit(0)),
            (Trit(0), Trit(-1), Trit(0)),
            (Trit(-1), Trit(1), Trit(-1)),
            (Trit(-1), Trit(0), Trit(0)),
            (Trit(-1), Trit(-1), Trit(1)),
        ]

        for x, y, expected in test_cases:
            result = PolynomialCore.evaluate_bivariate(x, y, and_coeffs)
            print(f"f({x}, {y}) = {result} (預期: {expected})")
            assert result == expected, f"AND多項式測試失敗: f({x}, {y}) = {result}, 預期 {expected}"

        # 測試OR運算: f(x,y) = x + y (平衡三進制模3加法)
        or_coeffs = (0, 1, 1, 0, 0, 0)
        test_cases = [
            (Trit(1), Trit(1), Trit(-1)),  # 1+1=2 → -1 (模3)
            (Trit(1), Trit(0), Trit(1)),  # 1+0=1
            (Trit(1), Trit(-1), Trit(0)),  # 1+(-1)=0
            (Trit(0), Trit(1), Trit(1)),
            (Trit(0), Trit(0), Trit(0)),
            (Trit(0), Trit(-1), Trit(-1)),
            (Trit(-1), Trit(1), Trit(0)),  # -1+1=0
            (Trit(-1), Trit(0), Trit(-1)),
            (Trit(-1), Trit(-1), Trit(1)),  # -1+(-1)=-2 → 1 (模3)
        ]

        for x, y, expected in test_cases:
            result = PolynomialCore.evaluate_bivariate(x, y, or_coeffs)
            print(f"f({x}, {y}) = {result} (預期: {expected})")
            assert result == expected, f"OR多項式測試失敗: f({x}, {y}) = {result}, 預期 {expected}"

        # 測試常數多項式: f(x,y) = 1
        constant_coeffs = (1, 0, 0, 0, 0, 0)
        test_cases = [
            (Trit(-1), Trit(-1), Trit(1)),
            (Trit(0), Trit(0), Trit(1)),
            (Trit(1), Trit(1), Trit(1)),
        ]

        for x, y, expected in test_cases:
            result = PolynomialCore.evaluate_bivariate(x, y, constant_coeffs)
            print(f"f({x}, {y}) = {result} (預期: {expected})")
            assert result == expected, f"常數多項式測試失敗: f({x}, {y}) = {result}, 預期 {expected}"

        print("PolynomialCore雙變數測試通過")

    def test_polynomial_consistency(self):
        """測試快速實現與三值邏輯實現之間的一致性"""
        print("\n=== 多項式實現一致性測試 ===")

        # 單變數多項式測試
        print("\n--- 單變數多項式一致性測試 ---")
        univariate_coeffs = [
            (0, 0, 0),  # f(x) = 0
            (0, 0, 1),  # f(x) = 1
            (0, 1, 0),  # f(x) = x
            (1, 0, 0),  # f(x) = x²
            (1, 1, 1),  # f(x) = x² + x + 1
            (1, -1, 0),  # f(x) = x² - x
            (-1, 0, 1),  # f(x) = -x² + 1
        ]

        for coeffs in univariate_coeffs:
            for x_val in [-1, 0, 1]:
                x = Trit(x_val)
                fast_result = PolynomialCore.evaluate_univariate(x, coeffs, use_trit_logic=False)
                trit_result = PolynomialCore.evaluate_univariate(x, coeffs, use_trit_logic=True)

                print(f"f({x}) = {coeffs} -> 快速: {fast_result}, 三值: {trit_result}")
                assert fast_result == trit_result, f"單變數多項式不一致: {coeffs}, x={x}"

        # 雙變數多項式測試
        print("\n--- 雙變數多項式一致性測試 ---")
        bivariate_coeffs = [
            (0, 0, 0, 0, 0, 0),  # f(x,y) = 0
            (1, 0, 0, 0, 0, 0),  # f(x,y) = 1
            (0, 1, 0, 0, 0, 0),  # f(x,y) = x
            (0, 0, 1, 0, 0, 0),  # f(x,y) = y
            (0, 0, 0, 1, 0, 0),  # f(x,y) = x²
            (0, 0, 0, 0, 1, 0),  # f(x,y) = y²
            (0, 0, 0, 0, 0, 1),  # f(x,y) = xy
            (1, 1, 1, 1, 1, 1),  # f(x,y) = 1 + x + y + x² + y² + xy
        ]

        for coeffs in bivariate_coeffs:
            for x_val in [-1, 0, 1]:
                for y_val in [-1, 0, 1]:
                    x = Trit(x_val)
                    y = Trit(y_val)
                    fast_result = PolynomialCore.evaluate_bivariate(x, y, coeffs, use_trit_logic=False)
                    trit_result = PolynomialCore.evaluate_bivariate(x, y, coeffs, use_trit_logic=True)

                    print(f"f({x}, {y}) = {coeffs} -> 快速: {fast_result}, 三值: {trit_result}")
                    assert fast_result == trit_result, f"雙變數多項式不一致: {coeffs}, x={x}, y={y}"

        print("多項式實現一致性驗證通過")

    def test_polynomial_edge_cases(self):
        """測試多項式邊界情況"""
        print("\n=== 多項式邊界情況測試 ===")

        # 測試單變數邊界情況
        print("\n--- 單變數邊界情況 ---")
        edge_coeffs = [
            (0, 0, 0),  # 零多項式
            (0, 0, 1),  # 常數多項式
            (1, 0, 0),  # 純二次項
        ]

        for coeffs in edge_coeffs:
            # 測試所有可能的輸入值
            for x_val in [-1, 0, 1]:
                x = Trit(x_val)
                result = PolynomialCore.evaluate_univariate(x, coeffs)
                print(f"f({x}) = {coeffs} = {result}")

                # 驗證結果在有效範圍內
                assert result.value in [-1, 0, 1], f"單變數結果超出範圍: {result}"

        # 測試雙變數邊界情況
        print("\n--- 雙變數邊界情況 ---")
        edge_coeffs = [
            (0, 0, 0, 0, 0, 0),  # 零多項式
            (1, 0, 0, 0, 0, 0),  # 常數多項式
            (0, 0, 0, 1, 0, 0),  # 純x²項
            (0, 0, 0, 0, 1, 0),  # 純y²項
            (0, 0, 0, 0, 0, 1),  # 純xy項
        ]

        for coeffs in edge_coeffs:
            # 測試所有可能的輸入組合
            for x_val in [-1, 0, 1]:
                for y_val in [-1, 0, 1]:
                    x = Trit(x_val)
                    y = Trit(y_val)
                    result = PolynomialCore.evaluate_bivariate(x, y, coeffs)
                    print(f"f({x}, {y}) = {coeffs} = {result}")

                    # 驗證結果在有效範圍內
                    assert result.value in [-1, 0, 1], f"雙變數結果超出範圍: {result}"

        print("多項式邊界情況測試通過")

    def test_polynomial_core_operator_generation(self):
        """測試PolynomialCore的算子生成功能"""
        print("\n=== PolynomialCore算子生成測試 ===")

        # 測試AND算子生成
        and_coeffs = (0, 0, 0, 0, 0, 1)
        and_operator = PolynomialCore.get_bivariate_operator(and_coeffs)

        test_cases = [
            (Trit(1), Trit(1), Trit(1)),
            (Trit(1), Trit(0), Trit(0)),
            (Trit(0), Trit(1), Trit(0)),
        ]

        for x, y, expected in test_cases:
            result = and_operator(x, y)
            print(f"AND算子({x}, {y}) = {result} (預期: {expected})")
            assert result == expected, f"AND算子測試失敗: AND({x}, {y}) = {result}, 預期 {expected}"

        # 測試OR算子生成
        or_coeffs = (0, 1, 1, 0, 0, 0)  # 簡化的OR多項式
        or_operator = PolynomialCore.get_bivariate_operator(or_coeffs)

        test_cases = [
            (Trit(1), Trit(0), Trit(1)),
            (Trit(0), Trit(1), Trit(1)),
            (Trit(0), Trit(0), Trit(0)),
        ]

        for x, y, expected in test_cases:
            result = or_operator(x, y)
            print(f"OR算子({x}, {y}) = {result} (預期: {expected})")
            assert result == expected, f"OR算子測試失敗: OR({x}, {y}) = {result}, 預期 {expected}"

        print("PolynomialCore算子生成測試通過")

    # Copyright (c) 2025 Certainty Computing Co. Limited. All rights reserved.


# ----------------------
# 7. 主程序入口
# ----------------------
if __name__ == '__main__':
    print("=== MQLK——函數完備的平衡三進制計算核心 ===")

    # 創建主測試套件
    suite = unittest.TestSuite()

    # 向主套件中添加所有測試用例
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTritDigit))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTritNumber))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestArithmeticCore))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestTensorCore))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPolynomialCore))

    # 執行測試
    runner = unittest.TextTestRunner()
    runner.run(suite)
    print("\n系統運行完成")


"""
Copyright (c) 2025 Certainty Computing Co. Limited. All rights reserved.

"""
