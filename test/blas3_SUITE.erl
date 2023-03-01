-module(blas3_SUITE).
-include_lib("eunit/include/eunit.hrl").


-record(instance, {size, name, expected}).
-record(blas, {signature, instances}).
-define(is_complex(S), (Size==c orelse Size == z)).

build_call(Signature, #instance{size = Size, name = Name}, Vars)->
    Variable_fetcher = fun(Arg_name) ->
        case orddict:find(Arg_name, Vars) of 

            {ok, Value} when is_list(Value) ->
                Split_ratio = if 
                    ?is_complex(Size) andalso (Arg_name == a orelse Arg_name == b orelse Arg_name == c) ->
                        1/2;
                    %?is_complex(Size) ->
                    %    1/2;
                    true ->
                        1
                end,
                Value_list = element(1, lists:split(round(length(Value)*Split_ratio), Value)),
                %io:format("Result: ~p~n", [Value_list]),
                blas:new(chain:ltb(Size, Value_list));

            {ok, Value} when (Arg_name == n) andalso ?is_complex(Size) ->
                floor(Value / 2);

            {ok, Value} -> 
                Value;

            _ when Arg_name == inc->
                1;

            _ ->
                Arg_name
        end
    end,
    [Name] ++ lists:map(Variable_fetcher, Signature).

compare({instance, Size, Name, _}, Lval, Rval)->
    Lval_comparable = if
        is_list(Rval) -> chain:btl(Size, blas:to_bin(Lval));
        true          -> Lval
    end,

    io:format("Result: ~p ~p ~p~n", [Name, Lval_comparable, Rval]),
    Lval_comparable == Rval.

run(Var_table, #blas{ signature = Signature, instances = Instances})->
    Test_instance = fun (Instance=#instance{expected = Expected}) ->
        Call = build_call(Signature, Instance, Var_table),
        %io:format("Calling ~p~n", [Call]),
        Out  = blas:run(list_to_tuple(Call)),
        State= orddict:from_list(lists:zip([out, name] ++ Signature, [Out] ++ Call)),
        true = lists:all(
            fun({Arg_name, Expected_val})->
                compare(Instance, orddict:fetch(Arg_name, State), Expected_val)
            end,
            Expected
        )
    end,

    lists:all(Test_instance, Instances).


ge_test()->
    Vars = orddict:from_list([
        {alpha, [1,0]},
        {beta,  [2,1]},
        {order, blasRowMajor},
        {transA, blasNoTrans},
        {transB, blasTrans},
        {n, 4},
        {a, [1, 3, 2, 4,  5, 3, 9, 2,  4, 1, 2,-2,  2, 2,-1, 1]},
        {b, [2, 2,-1, 1,  2, 2, 1, 5,  7, 2, 5, 0,  1, 3, 2, 4]},
        {c, [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]},

        {x, [1, -4, 2, 2]},
        {y, [1, -4, 1, 2]}
    ]),

    Fcts = [
        % gemm
        #blas{
            signature = [ order, transA, transB, n, n, n, alpha, a, n, b, n, beta, c, n],
            instances = [
              #instance{size = s, name = sgemm,   expected = [{c, [12, 30, 23, 30, 9, 37, 86, 40, 6, 2, 42, 3, 10, 12, 13, 12]}]},
              #instance{size = d, name = dgemm,   expected = [{c, [12, 30, 23, 30, 9, 37, 86, 40, 6, 2, 42, 3, 10, 12, 13, 12]}]},
              #instance{size = c, name = cgemm,   expected = [{c, [-8, 7, -22, 22, -8, 25, 3, 63]}]},
              #instance{size = z, name = zgemm,   expected = [{c, [-8, 7, -22, 22, -8, 25, 3, 63]}]},
              #instance{size = c, name = cgemm3m, expected = [{c, [-8, 7, -22, 22, -8, 25, 3, 63]}]},
              #instance{size = z, name = zgemm3m, expected = [{c, [-8, 7, -22, 22, -8, 25, 3, 63]}]}
            ]
        },
        %gemv
        #blas{
            signature = [ order, transA, n, n, alpha, a, n, x, inc, beta, y, inc],
            instances = [
              #instance{size = s, name = sgemv,   expected = [{y, [3.0,7.0,2.0,-2.0]}]},
              #instance{size = d, name = dgemv,   expected = [{y, [3.0,7.0,2.0,-2.0]}]},
              #instance{size = c, name = cgemv,   expected = [{y, [15.0,4.0,31.0,10.0]}]},
              #instance{size = z, name = zgemv,   expected = [{y, [15.0,4.0,31.0,10.0]}]}
            ]
        },
        %ger
        #blas{
            signature = [ order, n, n, alpha, x, inc, y, inc, a, n],
            instances = [
              #instance{size = s, name = sger,  expected = [{a, [2, -1, 3, 6, 1, 19, 5, -6, 6, -7, 4, 2, 4, -6, 1, 5]}]},
              #instance{size = d, name = dger,  expected = [{a, [2, -1, 3, 6, 1, 19, 5, -6, 6, -7, 4, 2, 4, -6, 1, 5]}]}
              #instance{size = c, name = cgeru, expected = [{a, [-14, -5, 11, 2, 15, -3, 7, 8]}]},
              #instance{size = z, name = zgeru, expected = [{a, [-14, -5, 11, 2, 15, -3, 7, 8]}]},
              #instance{size = c, name = cgerc, expected = [{a, [18, 3, -5, -2, -1, 13, 15, 0]}]},
              #instance{size = z, name = zgerc, expected = [{a, [18, 3, -5, -2, -1, 13, 15, 0]}]}
            ]
        }
    ],

    lists:all(fun (Blas) -> run(Vars, Blas) end, Fcts).


sp_test()->
     Vars = orddict:from_list([
        {alpha, [2,0]},
        {beta,  [3,1]},
        {uplo, blasUpper}, 
        {order, blasRowMajor},
        {n, 3},
        {a, [1, 3, 2, 
                3, 9,
                    2]},

        {x, [1, 2, -1]},
        {y, [0, -4, 1]}
     ]),

      Fcts = [
        % spmv
        #blas{
            signature = [ order, uplo, n, alpha, a, x, inc, beta, y, inc],
            instances = [
              #instance{size = s, name = sspmv,   expected = [{y, [10, -12, 39]}]},
              #instance{size = d, name = dspmv,   expected = [{y, [10, -12, 39]}]}
            ]
        },

        #blas{
            signature = [ order, uplo, n, alpha, x, inc, a],
            instances = [
              #instance{size = s, name = sspr,   expected = [{a, [3, 7, 0, 11, 5, 4]}]},
              #instance{size = d, name = dspr,   expected = [{a, [3, 7, 0, 11, 5, 4]}]}
            ]
        }, 

         #blas{
            signature = [ order, uplo, n, alpha, x, inc, y, inc, a],
            instances = [
              #instance{size = s, name = sspr2,   expected = [{a, [1, -5, 4, -29, 21, -2]}]},
              #instance{size = d, name = dspr2,   expected = [{a, [1, -5, 4, -29, 21, -2]}]}
            ]
        }
      ],

      
    lists:all(fun (Blas) -> run(Vars, Blas) end, Fcts). 

tr_test()->
     Vars = orddict:from_list([
        {order, blasRowMajor},
        {side, blasLeft},
        {uplo, blasUpper},
        {trans, blasNoTrans},
        {diag, blasNonUnit},
        {n, 4},
        {alpha, [1,-1]},
        {a, [1,0,3,0, 0,5,0,7, 0,0,9,0, 0,0,0,0]},
        {aa, [1,0,0,1, 0,0,1,0]},
        {b, [1,2,3,4, 0,1,6,7, 0,0,1,9, 0,0,0,1]},
        {x, [-1,2,-3,4]},
        {xx, [1,0,1,0]},
        {y, [1,2,3,4]},
        {a_trsm, [1,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,1]}
     ]),

      Fcts = [
        #blas{
            signature = [ order, uplo, trans, diag, n, a, n, x, inc],
            instances = [
                
                % trmv
                #instance{size = s, name = strmv, expected = [{x, [-10,38,-27,0]}]},
                #instance{size = d, name = dtrmv, expected = [{x, [-10,38,-27,0]}]},
                #instance{size = c, name = ctrmv, expected = [{x, [-10,14,-28,-21]}]},
                #instance{size = z, name = ztrmv, expected = [{x, [-10,14,-28,-21]}]}
            ]
        },
        #blas{
            signature = [ order, uplo, trans, diag, n, b, n, y, inc],
            instances = [
                 % trsv
                #instance{size = s, name = strsv, expected = [{y, [-260, 172, -33, 4]}]},
                #instance{size = d, name = dtrsv, expected = [{y, [-260, 172, -33, 4]}]}
            ]
        },
         #blas{
            signature = [ order, uplo, trans, diag, n, aa, n, xx, inc],
            instances = [
                 % trsv
                #instance{size = c, name = ctrsv, expected = [{xx, [1,-1,1,0]}]},
                #instance{size = z, name = ztrsv, expected = [{xx, [1,-1,1,0]}]}
            ]
        }
      ],
      
    lists:all(fun (Blas) -> run(Vars, Blas) end, Fcts). 

