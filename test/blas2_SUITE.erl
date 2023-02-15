-module(blas2_SUITE).
-include_lib("eunit/include/eunit.hrl").

-record(blas, {size, vars, expected}).

build_call(#blas{size=S, vars=Call}, Vars)->
    lists:map(
        fun(V)->
            Is_key = orddict:is_key(V, Vars),
            if Is_key ->
                Value = orddict:fetch(V, Vars),
                if 
                    is_list(Value) ->
                        blas:new(chain:ltb(S, Value));
                    V == n andalso (S==c orelse S==z) -> 
                        floor(Value / 2);
                    true ->
                        Value
                end;
            true ->
                if V == inc -> 1;
                true -> V end
            end
        end,
        Call
    ).

compare(#blas{size=S, vars=Vars}, Lval, Rval)->
    if is_list(Rval)->
        Llist = chain:btl(S, if is_tuple(Lval) -> blas:to_bin(Lval); true-> Lval end),
    
        io:format("Result: ~p ~p ~p~n", [lists:nth(1, Vars),Llist,  Rval]),
        Llist ==  Rval;
    true->
        io:format("Result: ~p ~p ~p~n", [lists:nth(1, Vars), Lval,  Rval]),
        Lval == Rval
    end.

run_test()->
    AllVars = orddict:from_list([
        {n, 4},
        {x, [1,-2,3,-4]},
        {x2, [2,2,2,2]},
        {y, [3,2,0,-2]},
        {a, [1,0]},
        {c, 0.0},
        {s, 1.0}
    ]),

    % Parameters are named in orddict vars. inc is always translated to 1. out is the output of the blas.
    Fcts = [
        % axpy
        #blas{size=s, vars=[saxpy, n, a, x, inc, y, inc], expected=[{y, [4, 0, 3, -6]}]},
        #blas{size=d, vars=[daxpy, n, a, x, inc, y, inc], expected=[{y, [4, 0, 3, -6]}]},
        #blas{size=c, vars=[caxpy, n, a, x, inc, y, inc], expected=[{y, [4, 0, 3, -6]}]},
        #blas{size=z, vars=[zaxpy, n, a, x, inc, y, inc], expected=[{y, [4, 0, 3, -6]}]},
        % copy
        #blas{size=s, vars=[scopy, n, x, inc, y, inc], expected=[{y, [1,-2,3,-4]}]},
        #blas{size=d, vars=[dcopy, n, x, inc, y, inc], expected=[{y, [1,-2,3,-4]}]},
        #blas{size=c, vars=[ccopy, n, x, inc, y, inc], expected=[{y, [1,-2,3,-4]}]},
        #blas{size=z, vars=[zcopy, n, x, inc, y, inc], expected=[{y, [1,-2,3,-4]}]},
        % swap
        #blas{size=s, vars=[sswap, n, x, inc, y, inc], expected=[{x, [3,2,0,-2]}, {y, [1,-2,3,-4]}]},
        #blas{size=d, vars=[dswap, n, x, inc, y, inc], expected=[{x, [3,2,0,-2]}, {y, [1,-2,3,-4]}]},
        #blas{size=c, vars=[cswap, n, x, inc, y, inc], expected=[{x, [3,2,0,-2]}, {y, [1,-2,3,-4]}]},
        #blas{size=z, vars=[zswap, n, x, inc, y, inc], expected=[{x, [3,2,0,-2]}, {y, [1,-2,3,-4]}]},
        % scal
        #blas{size=s, vars=[sscal,  n, a, x, inc], expected=[{x, [1,-2,3,-4]}]},
        #blas{size=d, vars=[dscal,  n, a, x, inc], expected=[{x, [1,-2,3,-4]}]},
        #blas{size=c, vars=[cscal,  n, a, x, inc], expected=[{x, [1,-2,3,-4]}]},
        #blas{size=z, vars=[zscal,  n, a, x, inc], expected=[{x, [1,-2,3,-4]}]},
        #blas{size=c, vars=[csscal, n, a, x, inc], expected=[{x, [1,-2,3,-4]}]},
        #blas{size=z, vars=[zdscal, n, a, x, inc], expected=[{x, [1,-2,3,-4]}]},
        % dot
        #blas{size=s, vars=[sdot,  n, x, inc, y, inc], expected=[{out, 7}]},
        #blas{size=d, vars=[ddot,  n, x, inc, y, inc], expected=[{out, 7}]},
        #blas{size=s, vars=[dsdot, n, x, inc, y, inc], expected=[{out, 7}]},
        #blas{size=c, vars=[cdotu, n, x, inc, y, inc], expected=[{out, [-1, -10]}]},
        #blas{size=z, vars=[zdotu, n, x, inc, y, inc], expected=[{out, [-1, -10]}]},
        #blas{size=c, vars=[cdotc, n, x, inc, y, inc], expected=[{out, [7, 2]}]},
        #blas{size=z, vars=[zdotc, n, x, inc, y, inc], expected=[{out, [7, 2]}]},
        #blas{size=s, vars=[sdsdot,n, a, x, inc, y, inc], expected=[{out, 8}]},
        % asum
        #blas{size=s, vars=[sasum,  n, x, inc], expected=[{out, 10}]},
        #blas{size=d, vars=[dasum,  n, x, inc], expected=[{out, 10}]},
        #blas{size=c, vars=[scasum, n, x, inc], expected=[{out, 10}]},
        #blas{size=z, vars=[dzasum, n, x, inc], expected=[{out, 10}]},
        % sum
        %#blas{size=s, vars=[ssum,  n, x, inc], expected=[{out, -2}]},
        %#blas{size=d, vars=[dsum,  n, x, inc], expected=[{out, -2}]},
        %#blas{size=c, vars=[scsum, n, x, inc], expected=[{out, -2}]},
        %#blas{size=z, vars=[dzsum, n, x, inc], expected=[{out, -2}]}
        % i_min removed
        % i_max removed
        % i_amin
        #blas{size=s, vars=[isamin,  n, x, inc], expected=[{out, 0}]},
        #blas{size=d, vars=[idamin,  n, x, inc], expected=[{out, 0}]},
        #blas{size=c, vars=[icamin, n, x, inc],  expected=[{out, 0}]},
        #blas{size=z, vars=[izamin, n, x, inc],  expected=[{out, 0}]},
        %i_amax
        #blas{size=s, vars=[isamax,  n, x, inc], expected=[{out, 3}]},
        #blas{size=d, vars=[idamax,  n, x, inc], expected=[{out, 3}]},
        #blas{size=c, vars=[icamax, n, x, inc],  expected=[{out, 1}]},
        #blas{size=z, vars=[izamax, n, x, inc],  expected=[{out, 1}]},
        %nrm
        #blas{size=s, vars=[snrm2,  n, x2, inc], expected=[{out, 4}]},
        #blas{size=d, vars=[dnrm2,  n, x2, inc], expected=[{out, 4}]},
        #blas{size=c, vars=[scnrm2, n, x2, inc], expected=[{out, 4}]},
        #blas{size=z, vars=[dznrm2, n, x2, inc], expected=[{out, 4}]},
        %rot
        #blas{size=s, vars=[srot, n, x, inc, y, inc, c, s], expected=[{x, [3,2,0,-2]}, {y, [-1,2,-3,4]}]},
        #blas{size=d, vars=[drot, n, x, inc, y, inc, c, s], expected=[{x, [3,2,0,-2]}, {y, [-1,2,-3,4]}]},
        #blas{size=c, vars=[csrot, n, x, inc, y, inc, c, s], expected=[{x, [3,2,0,-2]}, {y, [-1,2,-3,4]}]},
        #blas{size=z, vars=[zdrot, n, x, inc, y, inc, c, s], expected=[{x, [3,2,0,-2]}, {y, [-1,2,-3,4]}]}

    ],

    lists:all(
        fun(Blas=#blas{vars=BlasVars, expected=Expected})->
            Call = build_call(Blas, AllVars),
            io:format("Call: ~p~n", [Call]),
            Out = blas:run(list_to_tuple(Call)),
            Result = orddict:from_list(lists:zip([out] ++ BlasVars, [Out] ++ Call)),

            Passed = lists:all(
                fun({Arg, Val})->
                    compare(Blas, orddict:fetch(Arg, Result), Val)
                end,
                Expected
            ),

            if not Passed ->
                nok = lists:nth(1, BlasVars);
            true ->
                true
            end
        end,
        Fcts
    ).