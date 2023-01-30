-module(chain).

-export([ltb/2, btl/2]).

-define(IS_PAIR_LENGTH(List), (length(List) band 1) == 0 ).
-define(IS_SIZE_ALIGNED(Binary, ElementSize), size(Binary) rem ElementSize == 0).
-define(IS_PAIR_ELEMENTS(Binary, ElementSize), (round(size(Binary)/ElementSize) band 1) == 0).

ltb(Type, List)->
    case Type of 
        s -> << <<V:32/native-float>> || V <- List >>;
        d -> << <<V:64/native-float>> || V <- List >>;
        c when ?IS_PAIR_LENGTH(List) -> ltb(s, List); % Two elements per items
        z when ?IS_PAIR_LENGTH(List) -> ltb(d, List)  % Two elements per items
    end.

btl(Type, Binary)->
    case Type of 
        s when ?IS_SIZE_ALIGNED(Binary, 4)  -> [ V || <<V:32/native-float>> <= Binary ];
        d when ?IS_SIZE_ALIGNED(Binary, 8)  -> [ V || <<V:64/native-float>> <= Binary ];
        c when ?IS_PAIR_ELEMENTS(Binary, 4) -> btl(s, Binary);
        z when ?IS_PAIR_ELEMENTS(Binary, 8) -> btl(d, Binary)
    end.