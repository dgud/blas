

typedef enum sizes {s_bytes=4, d_bytes=8, c_bytes=8, z_bytes=16, no_bytes=0} size_in_bytes;

typedef enum BLAS_NAMES {
    saxpy=210727551034,
    daxpy=210709762219,
    caxpy=210708576298,
    zaxpy=210735852481,
    scopy=210727613107,
    dcopy=210709824292,
    ccopy=210708638371,
    zcopy=210735914554,
    sswap=210728196307,
    dswap=210710407492,
    cswap=210709221571,
    zswap=210736497754,
    sscal=210728174523,
    dscal=210710385708,
    cscal=210709199787,
    csscal=6953404169886,
    zscal=210736475970,
    zdscal=6954286495110,
    sdot=6385686335,
    ddot=6385147280,
    cdotu=210708674436,
    zdotu=210735950619,
    cdotc=210708674418,
    zdotc=210735950601,
    dsdot=210710387267,
    sdsdot=6954012548918,
    snrm2=210728011511,
    dnrm2=210710222696,
    scnrm2=6954011198426,
    dznrm2=6953451443714,
    sasum=210727545742,
    dasum=210709756927,
    scasum=6954010732657,
    dzasum=6953450977945,
    isamax=6953638346280,
    idamax=6953620557465,
    icamax=6953619371544,
    izamax=6953646647727,
    srot=6385701581,
    drot=6385162526,
    csrot=210709216592,
    zdrot=210735953720,
    srotg=210728152276,
    drotg=210710363461,
    crotg=210709177540,
    zrotg=210736453723,
    srotmg=6954029025409,
    drotmg=6953441994514,
    srotm=210728152282,
    drotm=210710363467, 
    isamin=6953638346534,
    idamin=6953620557719,
    icamin=6953619371798,
    izamin=6953646647981,
    ismax=210716326215,
    idmax=210715787160,
    icmax=210715751223,
    izmax=210716577774,
    ismin=210716326469,
    idmin=210715787414,
    icmin=210715751477,
    izmin=210716578028,
    sgemv=210727745863,
    dgemv=210709957048,
    cgemv=210708771127,
    zgemv=210736047310,
    sgbmv=210727742596,
    dgbmv=210709953781,
    cgbmv=210708767860,
    zgbmv=210736044043,
    ssbmv=210728173840,
    dsbmv=210710385025,
    sger=6385689270,
    dger=6385150215,
    strmv=210728227201,
    dtrmv=210710438386,
    ctrmv=210709252465,
    ztrmv=210736528648,
    strsv=210728227399,
    dtrsv=210710438584,
    ctrsv=6385128901,
    ztrsv=210736528846,
    strsm=210728227390,
    dtrsm=210710438575,
    ctrsm=210709252654,
    ztrsm=210736528837,
    cgeru=210708771291,
    cgerc=210708771273,
    zgeru=210736047474,
    zgerc=210736047456,
    sgemm=210727745854,
    dgemm=210709957039,
    cgemm=210708771118,
    cgemm3m=229461851749294,
    zgemm=210736047301,
    zgemm3m=229491555512581,
    stbmv=210728209777,
    dtbmv=210710420962,
    ctbmv=210709235041,
    ztbmv=210736511224,
    stbsv=210728209975,
    dtbsv=210710421160,
    ctbsv=210709235239,
    ztbsv=210736511422,
    stpmv=210728225023,
    dtpmv=210710436208,
    ctpmv=210709250287,
    ztpmv=210736526470,
    stpsv=210728225221,
    dtpsv=210710436406,
    ctpsv=210709250485,
    ztpsv=210736526668,
    ssymv=210728198887, 
    dsymv=6385170085,  // incorrect
    chemv=210708807064,
    zhemv=210736083247,
    sspmv=210728189086,
    dspmv=210710400271,
    sspr=6385702701,
    dspr=6385163646,
    chpr=6385115730,
    zhpr=6385942281,
    sspr2=210728189183,
    dspr2=210710400368,
    chpr2=210708819140,
    zhpr2=210736095323,
    chbmv=210708803797,
    zhbmv=210736079980,
    chpmv=210708819043,
    zhpmv=210736095226,
    cher=6385115367,
    zher=6385941918,
    chemm=210708807055,
    zhemm=210736083238,
    cherk=210708807218,
    zherk=210736083401,
    cher2k=6953390636420,
    zher2k=6954290750459,
    ssymm=210728198878,
    dsymm=210710410063,
    csymm=210709224142,
    zsymm=210736500325,
    ssyrk=210728199041,
    dsyrk=210710410226,
    csyrk=210709224305,
    zsyrk=210736500488,
    ssyr2k=6954030566579,
    dsyr2k=6953443535684,
    csyr2k=6953404400291,
    zsyr2k=6954304514330,
    ssum=6385702861,
    dsum=6385163806,
    dzsum=210710655352,
    scsum=210727617616,
    cher2=210708807161,
    zher2=210736083344,
    strmm=210728227192,
    dtrmm=210710438377,
    ctrmm=210709252456,
    ztrmm=210736528639,
    ssyr=6385702998,
    dsyr=6385163943,
    ssyr2=210728198984,
    dsyr2=210710410169,
    blas_name_end=0
} blas_names;

size_in_bytes pick_size(long unsigned hash);
size_in_bytes pick_size(long unsigned hash){
    size_in_bytes type;

     switch(hash){
        case saxpy: case scopy: case sswap: case sscal: case sdot: case srot: 
        case srotmg:case sgemv: case sger:  case sgbmv: case ssbmv:case srotg: 
        case snrm2: case sasum: case isamax:case isamin:case ismax:case ismin:
        case srotm: case sdsdot: case dsdot: case ssum:
        case strmv: case strmm: case strsv: case strsm:
        case sgemm: case sspmv: case sspr: case sspr2:
            type = s_bytes;  
        break;   

        case daxpy: case dcopy: case dswap:  case dscal: case ddot: case drot:
        case drotg: case drotm: case drotmg: case dgemv: case dger: case dgbmv: 
        case dnrm2: case dasum: case idamax: case idamin:case idmax:case idmin: 
        case dsbmv: case dsum: case csrot:
        case dtrmv: case dtrsv: case dtrmm: case dtrsm:
        case dgemm: case dspmv: case dspr: case dspr2:
            type = d_bytes;  
        break;

        case caxpy: case ccopy: case cscal: case cdotu: case cgemv: case cgeru: 
        case scnrm2:case scasum:case icamax:case icamin:case icmax: case icmin: 
        case cgerc: case cgbmv: case cswap: case csscal: case cdotc:
        case ctrmv: case ctrsv: case ctrmm: case ctrsm: case scsum:
        case cgemm: case cgemm3m:
            type = c_bytes;  
        break;

        case zaxpy: case zcopy: case zswap: case zscal: case zdotu: case zdotc: 
        case dznrm2:case dzasum:case izamax:case izamin:case izmax: case izmin: 
        case zrotg: case zgemv: case zgeru: case zgerc: case zgbmv: case dzsum:
        case ztrmv: case ztrsv: case ztrmm: case ztrsm: case zdscal: case zdrot:
        case zgemm: case zgemm3m:
            type = z_bytes;  
        break;
        
        default:
            type = no_bytes;
        break;
    }

    return type;
}