ЭЌ
ѕ
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02unknown8рГ
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
|
dense_308/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_308/kernel
u
$dense_308/kernel/Read/ReadVariableOpReadVariableOpdense_308/kernel*
_output_shapes

:*
dtype0
t
dense_308/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_308/bias
m
"dense_308/bias/Read/ReadVariableOpReadVariableOpdense_308/bias*
_output_shapes
:*
dtype0
|
dense_309/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_309/kernel
u
$dense_309/kernel/Read/ReadVariableOpReadVariableOpdense_309/kernel*
_output_shapes

:*
dtype0
t
dense_309/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_309/bias
m
"dense_309/bias/Read/ReadVariableOpReadVariableOpdense_309/bias*
_output_shapes
:*
dtype0
|
dense_310/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_310/kernel
u
$dense_310/kernel/Read/ReadVariableOpReadVariableOpdense_310/kernel*
_output_shapes

:*
dtype0
t
dense_310/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_310/bias
m
"dense_310/bias/Read/ReadVariableOpReadVariableOpdense_310/bias*
_output_shapes
:*
dtype0
|
dense_311/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_311/kernel
u
$dense_311/kernel/Read/ReadVariableOpReadVariableOpdense_311/kernel*
_output_shapes

:*
dtype0
t
dense_311/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_311/bias
m
"dense_311/bias/Read/ReadVariableOpReadVariableOpdense_311/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

Adam/dense_308/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_308/kernel/m

+Adam/dense_308/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_308/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_308/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_308/bias/m
{
)Adam/dense_308/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_308/bias/m*
_output_shapes
:*
dtype0

Adam/dense_309/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_309/kernel/m

+Adam/dense_309/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_309/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_309/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_309/bias/m
{
)Adam/dense_309/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_309/bias/m*
_output_shapes
:*
dtype0

Adam/dense_310/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_310/kernel/m

+Adam/dense_310/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_310/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_310/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_310/bias/m
{
)Adam/dense_310/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_310/bias/m*
_output_shapes
:*
dtype0

Adam/dense_311/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_311/kernel/m

+Adam/dense_311/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_311/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_311/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_311/bias/m
{
)Adam/dense_311/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_311/bias/m*
_output_shapes
:*
dtype0

Adam/dense_308/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_308/kernel/v

+Adam/dense_308/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_308/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_308/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_308/bias/v
{
)Adam/dense_308/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_308/bias/v*
_output_shapes
:*
dtype0

Adam/dense_309/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_309/kernel/v

+Adam/dense_309/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_309/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_309/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_309/bias/v
{
)Adam/dense_309/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_309/bias/v*
_output_shapes
:*
dtype0

Adam/dense_310/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_310/kernel/v

+Adam/dense_310/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_310/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_310/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_310/bias/v
{
)Adam/dense_310/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_310/bias/v*
_output_shapes
:*
dtype0

Adam/dense_311/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_311/kernel/v

+Adam/dense_311/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_311/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_311/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_311/bias/v
{
)Adam/dense_311/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_311/bias/v*
_output_shapes
:*
dtype0
І
ConstConst*
_output_shapes

:*
dtype0*i
value`B^"P№p
BчBЗ3Bв&
BсЛEIѓб:    Њ=_МS`КAЊ=PЏ=+o=БP=х~=O=зх~=n~=Аg~=*~=
Ј
Const_1Const*
_output_shapes

:*
dtype0*i
value`B^"PC/Cг;CллCэЌSW:9    Тоѓ:є"`lNbВEТоѓ:xѕ:Ќѓ:И
ѕ:т3ѓ:ёЃі:дѓ:сэ№:ЂNё:Јя:

NoOpNoOp
А1
Const_2Const"/device:CPU:0*
_output_shapes
: *
dtype0*щ0
valueп0Bм0 Bе0
Д
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
Ѕ

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
h

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
Н
,iter

-beta_1

.beta_2
	/decaymTmUmVmW mX!mY&mZ'm[v\v]v^v_ v`!va&vb'vc
N
0
1
2
3
4
5
6
 7
!8
&9
'10
8
0
1
2
3
 4
!5
&6
'7
 
­
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
	regularization_losses
 
 
 
 
 
NL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcount5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE
 
\Z
VARIABLE_VALUEdense_308/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_308/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEdense_309/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_309/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
\Z
VARIABLE_VALUEdense_310/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_310/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
"	variables
#trainable_variables
$regularization_losses
\Z
VARIABLE_VALUEdense_311/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_311/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
(	variables
)trainable_variables
*regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
#
0
1
2
3
4

I0
J1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ktotal
	Lcount
M	variables
N	keras_api
D
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

M	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

O0
P1

R	variables
}
VARIABLE_VALUEAdam/dense_308/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_308/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_309/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_309/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_310/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_310/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_311/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_311/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_308/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_308/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_309/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_309/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_310/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_310/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_311/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_311/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

&serving_default_normalization_25_inputPlaceholder*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*%
shape:џџџџџџџџџџџџџџџџџџ
я
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_25_inputConstConst_1dense_308/kerneldense_308/biasdense_309/kerneldense_309/biasdense_310/kerneldense_310/biasdense_311/kerneldense_311/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_1621345
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ђ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_308/kernel/Read/ReadVariableOp"dense_308/bias/Read/ReadVariableOp$dense_309/kernel/Read/ReadVariableOp"dense_309/bias/Read/ReadVariableOp$dense_310/kernel/Read/ReadVariableOp"dense_310/bias/Read/ReadVariableOp$dense_311/kernel/Read/ReadVariableOp"dense_311/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/dense_308/kernel/m/Read/ReadVariableOp)Adam/dense_308/bias/m/Read/ReadVariableOp+Adam/dense_309/kernel/m/Read/ReadVariableOp)Adam/dense_309/bias/m/Read/ReadVariableOp+Adam/dense_310/kernel/m/Read/ReadVariableOp)Adam/dense_310/bias/m/Read/ReadVariableOp+Adam/dense_311/kernel/m/Read/ReadVariableOp)Adam/dense_311/bias/m/Read/ReadVariableOp+Adam/dense_308/kernel/v/Read/ReadVariableOp)Adam/dense_308/bias/v/Read/ReadVariableOp+Adam/dense_309/kernel/v/Read/ReadVariableOp)Adam/dense_309/bias/v/Read/ReadVariableOp+Adam/dense_310/kernel/v/Read/ReadVariableOp)Adam/dense_310/bias/v/Read/ReadVariableOp+Adam/dense_311/kernel/v/Read/ReadVariableOp)Adam/dense_311/bias/v/Read/ReadVariableOpConst_2*0
Tin)
'2%		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__traced_save_1621680
Џ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_308/kerneldense_308/biasdense_309/kerneldense_309/biasdense_310/kerneldense_310/biasdense_311/kerneldense_311/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1total_1count_2Adam/dense_308/kernel/mAdam/dense_308/bias/mAdam/dense_309/kernel/mAdam/dense_309/bias/mAdam/dense_310/kernel/mAdam/dense_310/bias/mAdam/dense_311/kernel/mAdam/dense_311/bias/mAdam/dense_308/kernel/vAdam/dense_308/bias/vAdam/dense_309/kernel/vAdam/dense_309/bias/vAdam/dense_310/kernel/vAdam/dense_310/bias/vAdam/dense_311/kernel/vAdam/dense_311/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__traced_restore_1621795БЄ
Щ	
ї
F__inference_dense_311_layer_call_and_return_conditional_losses_1621076

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
З
и
#__inference__traced_restore_1621795
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_308_kernel:/
!assignvariableop_4_dense_308_bias:5
#assignvariableop_5_dense_309_kernel:/
!assignvariableop_6_dense_309_bias:5
#assignvariableop_7_dense_310_kernel:/
!assignvariableop_8_dense_310_bias:5
#assignvariableop_9_dense_311_kernel:0
"assignvariableop_10_dense_311_bias:'
assignvariableop_11_adam_iter:	 )
assignvariableop_12_adam_beta_1: )
assignvariableop_13_adam_beta_2: (
assignvariableop_14_adam_decay: #
assignvariableop_15_total: %
assignvariableop_16_count_1: %
assignvariableop_17_total_1: %
assignvariableop_18_count_2: =
+assignvariableop_19_adam_dense_308_kernel_m:7
)assignvariableop_20_adam_dense_308_bias_m:=
+assignvariableop_21_adam_dense_309_kernel_m:7
)assignvariableop_22_adam_dense_309_bias_m:=
+assignvariableop_23_adam_dense_310_kernel_m:7
)assignvariableop_24_adam_dense_310_bias_m:=
+assignvariableop_25_adam_dense_311_kernel_m:7
)assignvariableop_26_adam_dense_311_bias_m:=
+assignvariableop_27_adam_dense_308_kernel_v:7
)assignvariableop_28_adam_dense_308_bias_v:=
+assignvariableop_29_adam_dense_309_kernel_v:7
)assignvariableop_30_adam_dense_309_bias_v:=
+assignvariableop_31_adam_dense_310_kernel_v:7
)assignvariableop_32_adam_dense_310_bias_v:=
+assignvariableop_33_adam_dense_311_kernel_v:7
)assignvariableop_34_adam_dense_311_bias_v:
identity_36ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ѕ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Ы
valueСBО$B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHИ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B е
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*І
_output_shapes
::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_308_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_308_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_309_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_309_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_310_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_310_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_311_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_311_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_adam_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_308_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_308_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_309_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_309_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_310_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_310_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_311_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_311_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_308_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_308_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_309_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_309_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_310_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_310_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_311_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_311_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 б
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_36IdentityIdentity_35:output:0^NoOp_1*
T0*
_output_shapes
: О
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_36Identity_36:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ѕ

м
/__inference_sequential_77_layer_call_fn_1621370

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621083o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
ѓ

т
%__inference_signature_wrapper_1621345
normalization_25_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityЂStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallnormalization_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_1621001o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
0
_user_specified_namenormalization_25_input:$ 

_output_shapes

::$ 

_output_shapes

:
Щ

+__inference_dense_308_layer_call_fn_1621480

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_308_layer_call_and_return_conditional_losses_1621026o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ы
Ъ
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621083

inputs
normalization_25_sub_y
normalization_25_sqrt_x#
dense_308_1621027:
dense_308_1621029:#
dense_309_1621044:
dense_309_1621046:#
dense_310_1621061:
dense_310_1621063:#
dense_311_1621077:
dense_311_1621079:
identityЂ!dense_308/StatefulPartitionedCallЂ!dense_309/StatefulPartitionedCallЂ!dense_310/StatefulPartitionedCallЂ!dense_311/StatefulPartitionedCallm
normalization_25/subSubinputsnormalization_25_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_25/SqrtSqrtnormalization_25_sqrt_x*
T0*
_output_shapes

:_
normalization_25/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_25/MaximumMaximumnormalization_25/Sqrt:y:0#normalization_25/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_25/truedivRealDivnormalization_25/sub:z:0normalization_25/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
!dense_308/StatefulPartitionedCallStatefulPartitionedCallnormalization_25/truediv:z:0dense_308_1621027dense_308_1621029*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_308_layer_call_and_return_conditional_losses_1621026
!dense_309/StatefulPartitionedCallStatefulPartitionedCall*dense_308/StatefulPartitionedCall:output:0dense_309_1621044dense_309_1621046*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_309_layer_call_and_return_conditional_losses_1621043
!dense_310/StatefulPartitionedCallStatefulPartitionedCall*dense_309/StatefulPartitionedCall:output:0dense_310_1621061dense_310_1621063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_310_layer_call_and_return_conditional_losses_1621060
!dense_311/StatefulPartitionedCallStatefulPartitionedCall*dense_310/StatefulPartitionedCall:output:0dense_311_1621077dense_311_1621079*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_311_layer_call_and_return_conditional_losses_1621076y
IdentityIdentity*dense_311/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^dense_308/StatefulPartitionedCall"^dense_309/StatefulPartitionedCall"^dense_310/StatefulPartitionedCall"^dense_311/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2F
!dense_308/StatefulPartitionedCall!dense_308/StatefulPartitionedCall2F
!dense_309/StatefulPartitionedCall!dense_309/StatefulPartitionedCall2F
!dense_310/StatefulPartitionedCall!dense_310/StatefulPartitionedCall2F
!dense_311/StatefulPartitionedCall!dense_311/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Щ

+__inference_dense_311_layer_call_fn_1621540

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_311_layer_call_and_return_conditional_losses_1621076o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ы
Ъ
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621204

inputs
normalization_25_sub_y
normalization_25_sqrt_x#
dense_308_1621183:
dense_308_1621185:#
dense_309_1621188:
dense_309_1621190:#
dense_310_1621193:
dense_310_1621195:#
dense_311_1621198:
dense_311_1621200:
identityЂ!dense_308/StatefulPartitionedCallЂ!dense_309/StatefulPartitionedCallЂ!dense_310/StatefulPartitionedCallЂ!dense_311/StatefulPartitionedCallm
normalization_25/subSubinputsnormalization_25_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_25/SqrtSqrtnormalization_25_sqrt_x*
T0*
_output_shapes

:_
normalization_25/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_25/MaximumMaximumnormalization_25/Sqrt:y:0#normalization_25/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_25/truedivRealDivnormalization_25/sub:z:0normalization_25/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
!dense_308/StatefulPartitionedCallStatefulPartitionedCallnormalization_25/truediv:z:0dense_308_1621183dense_308_1621185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_308_layer_call_and_return_conditional_losses_1621026
!dense_309/StatefulPartitionedCallStatefulPartitionedCall*dense_308/StatefulPartitionedCall:output:0dense_309_1621188dense_309_1621190*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_309_layer_call_and_return_conditional_losses_1621043
!dense_310/StatefulPartitionedCallStatefulPartitionedCall*dense_309/StatefulPartitionedCall:output:0dense_310_1621193dense_310_1621195*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_310_layer_call_and_return_conditional_losses_1621060
!dense_311/StatefulPartitionedCallStatefulPartitionedCall*dense_310/StatefulPartitionedCall:output:0dense_311_1621198dense_311_1621200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_311_layer_call_and_return_conditional_losses_1621076y
IdentityIdentity*dense_311/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^dense_308/StatefulPartitionedCall"^dense_309/StatefulPartitionedCall"^dense_310/StatefulPartitionedCall"^dense_311/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2F
!dense_308/StatefulPartitionedCall!dense_308/StatefulPartitionedCall2F
!dense_309/StatefulPartitionedCall!dense_309/StatefulPartitionedCall2F
!dense_310/StatefulPartitionedCall!dense_310/StatefulPartitionedCall2F
!dense_311/StatefulPartitionedCall!dense_311/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Ч*

J__inference_sequential_77_layer_call_and_return_conditional_losses_1621471

inputs
normalization_25_sub_y
normalization_25_sqrt_x:
(dense_308_matmul_readvariableop_resource:7
)dense_308_biasadd_readvariableop_resource::
(dense_309_matmul_readvariableop_resource:7
)dense_309_biasadd_readvariableop_resource::
(dense_310_matmul_readvariableop_resource:7
)dense_310_biasadd_readvariableop_resource::
(dense_311_matmul_readvariableop_resource:7
)dense_311_biasadd_readvariableop_resource:
identityЂ dense_308/BiasAdd/ReadVariableOpЂdense_308/MatMul/ReadVariableOpЂ dense_309/BiasAdd/ReadVariableOpЂdense_309/MatMul/ReadVariableOpЂ dense_310/BiasAdd/ReadVariableOpЂdense_310/MatMul/ReadVariableOpЂ dense_311/BiasAdd/ReadVariableOpЂdense_311/MatMul/ReadVariableOpm
normalization_25/subSubinputsnormalization_25_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_25/SqrtSqrtnormalization_25_sqrt_x*
T0*
_output_shapes

:_
normalization_25/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_25/MaximumMaximumnormalization_25/Sqrt:y:0#normalization_25/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_25/truedivRealDivnormalization_25/sub:z:0normalization_25/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_308/MatMul/ReadVariableOpReadVariableOp(dense_308_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_308/MatMulMatMulnormalization_25/truediv:z:0'dense_308/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_308/BiasAdd/ReadVariableOpReadVariableOp)dense_308_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_308/BiasAddBiasAdddense_308/MatMul:product:0(dense_308/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_308/ReluReludense_308/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_309/MatMul/ReadVariableOpReadVariableOp(dense_309_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_309/MatMulMatMuldense_308/Relu:activations:0'dense_309/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_309/BiasAdd/ReadVariableOpReadVariableOp)dense_309_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_309/BiasAddBiasAdddense_309/MatMul:product:0(dense_309/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_309/ReluReludense_309/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_310/MatMul/ReadVariableOpReadVariableOp(dense_310_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_310/MatMulMatMuldense_309/Relu:activations:0'dense_310/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_310/BiasAdd/ReadVariableOpReadVariableOp)dense_310_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_310/BiasAddBiasAdddense_310/MatMul:product:0(dense_310/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_310/ReluReludense_310/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_311/MatMul/ReadVariableOpReadVariableOp(dense_311_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_311/MatMulMatMuldense_310/Relu:activations:0'dense_311/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_311/BiasAdd/ReadVariableOpReadVariableOp)dense_311_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_311/BiasAddBiasAdddense_311/MatMul:product:0(dense_311/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџi
IdentityIdentitydense_311/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџк
NoOpNoOp!^dense_308/BiasAdd/ReadVariableOp ^dense_308/MatMul/ReadVariableOp!^dense_309/BiasAdd/ReadVariableOp ^dense_309/MatMul/ReadVariableOp!^dense_310/BiasAdd/ReadVariableOp ^dense_310/MatMul/ReadVariableOp!^dense_311/BiasAdd/ReadVariableOp ^dense_311/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2D
 dense_308/BiasAdd/ReadVariableOp dense_308/BiasAdd/ReadVariableOp2B
dense_308/MatMul/ReadVariableOpdense_308/MatMul/ReadVariableOp2D
 dense_309/BiasAdd/ReadVariableOp dense_309/BiasAdd/ReadVariableOp2B
dense_309/MatMul/ReadVariableOpdense_309/MatMul/ReadVariableOp2D
 dense_310/BiasAdd/ReadVariableOp dense_310/BiasAdd/ReadVariableOp2B
dense_310/MatMul/ReadVariableOpdense_310/MatMul/ReadVariableOp2D
 dense_311/BiasAdd/ReadVariableOp dense_311/BiasAdd/ReadVariableOp2B
dense_311/MatMul/ReadVariableOpdense_311/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
К'
г
__inference_adapt_step_1616043
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂIteratorGetNextЂReadVariableOpЂReadVariableOp_1ЂReadVariableOp_2Ђadd/ReadVariableOpБ
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*'
_output_shapes
:џџџџџџџџџ*&
output_shapes
:џџџџџџџџџ*
output_types
2h
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*'
_output_shapes
:џџџџџџџџџl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 a
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	Z
GatherV2/indicesConst*
_output_shapes
:*
dtype0*
valueB: O
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:O
ConstConst*
_output_shapes
:*
dtype0*
valueB: P
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: f
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	X
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: K
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: G
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: I
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: J
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?H
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0P
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:X
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:G
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:d
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0V
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @J
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:f
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype0V
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:E
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:V
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @N
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:Z
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:I
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:I
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype0
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype0
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
ѕ

м
/__inference_sequential_77_layer_call_fn_1621395

inputs
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621204o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Щ

+__inference_dense_310_layer_call_fn_1621520

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_310_layer_call_and_return_conditional_losses_1621060o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ

+__inference_dense_309_layer_call_fn_1621500

inputs
unknown:
	unknown_0:
identityЂStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_309_layer_call_and_return_conditional_losses_1621043o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ї
F__inference_dense_310_layer_call_and_return_conditional_losses_1621060

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ї
F__inference_dense_308_layer_call_and_return_conditional_losses_1621026

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ћ
к
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621283
normalization_25_input
normalization_25_sub_y
normalization_25_sqrt_x#
dense_308_1621262:
dense_308_1621264:#
dense_309_1621267:
dense_309_1621269:#
dense_310_1621272:
dense_310_1621274:#
dense_311_1621277:
dense_311_1621279:
identityЂ!dense_308/StatefulPartitionedCallЂ!dense_309/StatefulPartitionedCallЂ!dense_310/StatefulPartitionedCallЂ!dense_311/StatefulPartitionedCall}
normalization_25/subSubnormalization_25_inputnormalization_25_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_25/SqrtSqrtnormalization_25_sqrt_x*
T0*
_output_shapes

:_
normalization_25/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_25/MaximumMaximumnormalization_25/Sqrt:y:0#normalization_25/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_25/truedivRealDivnormalization_25/sub:z:0normalization_25/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
!dense_308/StatefulPartitionedCallStatefulPartitionedCallnormalization_25/truediv:z:0dense_308_1621262dense_308_1621264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_308_layer_call_and_return_conditional_losses_1621026
!dense_309/StatefulPartitionedCallStatefulPartitionedCall*dense_308/StatefulPartitionedCall:output:0dense_309_1621267dense_309_1621269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_309_layer_call_and_return_conditional_losses_1621043
!dense_310/StatefulPartitionedCallStatefulPartitionedCall*dense_309/StatefulPartitionedCall:output:0dense_310_1621272dense_310_1621274*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_310_layer_call_and_return_conditional_losses_1621060
!dense_311/StatefulPartitionedCallStatefulPartitionedCall*dense_310/StatefulPartitionedCall:output:0dense_311_1621277dense_311_1621279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_311_layer_call_and_return_conditional_losses_1621076y
IdentityIdentity*dense_311/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^dense_308/StatefulPartitionedCall"^dense_309/StatefulPartitionedCall"^dense_310/StatefulPartitionedCall"^dense_311/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2F
!dense_308/StatefulPartitionedCall!dense_308/StatefulPartitionedCall2F
!dense_309/StatefulPartitionedCall!dense_309/StatefulPartitionedCall2F
!dense_310/StatefulPartitionedCall!dense_310/StatefulPartitionedCall2F
!dense_311/StatefulPartitionedCall!dense_311/StatefulPartitionedCall:h d
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
0
_user_specified_namenormalization_25_input:$ 

_output_shapes

::$ 

_output_shapes

:


ї
F__inference_dense_308_layer_call_and_return_conditional_losses_1621491

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Щ	
ї
F__inference_dense_311_layer_call_and_return_conditional_losses_1621550

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ї
F__inference_dense_309_layer_call_and_return_conditional_losses_1621043

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs


ї
F__inference_dense_310_layer_call_and_return_conditional_losses_1621531

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ћ
к
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621314
normalization_25_input
normalization_25_sub_y
normalization_25_sqrt_x#
dense_308_1621293:
dense_308_1621295:#
dense_309_1621298:
dense_309_1621300:#
dense_310_1621303:
dense_310_1621305:#
dense_311_1621308:
dense_311_1621310:
identityЂ!dense_308/StatefulPartitionedCallЂ!dense_309/StatefulPartitionedCallЂ!dense_310/StatefulPartitionedCallЂ!dense_311/StatefulPartitionedCall}
normalization_25/subSubnormalization_25_inputnormalization_25_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_25/SqrtSqrtnormalization_25_sqrt_x*
T0*
_output_shapes

:_
normalization_25/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_25/MaximumMaximumnormalization_25/Sqrt:y:0#normalization_25/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_25/truedivRealDivnormalization_25/sub:z:0normalization_25/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
!dense_308/StatefulPartitionedCallStatefulPartitionedCallnormalization_25/truediv:z:0dense_308_1621293dense_308_1621295*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_308_layer_call_and_return_conditional_losses_1621026
!dense_309/StatefulPartitionedCallStatefulPartitionedCall*dense_308/StatefulPartitionedCall:output:0dense_309_1621298dense_309_1621300*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_309_layer_call_and_return_conditional_losses_1621043
!dense_310/StatefulPartitionedCallStatefulPartitionedCall*dense_309/StatefulPartitionedCall:output:0dense_310_1621303dense_310_1621305*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_310_layer_call_and_return_conditional_losses_1621060
!dense_311/StatefulPartitionedCallStatefulPartitionedCall*dense_310/StatefulPartitionedCall:output:0dense_311_1621308dense_311_1621310*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_311_layer_call_and_return_conditional_losses_1621076y
IdentityIdentity*dense_311/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^dense_308/StatefulPartitionedCall"^dense_309/StatefulPartitionedCall"^dense_310/StatefulPartitionedCall"^dense_311/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2F
!dense_308/StatefulPartitionedCall!dense_308/StatefulPartitionedCall2F
!dense_309/StatefulPartitionedCall!dense_309/StatefulPartitionedCall2F
!dense_310/StatefulPartitionedCall!dense_310/StatefulPartitionedCall2F
!dense_311/StatefulPartitionedCall!dense_311/StatefulPartitionedCall:h d
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
0
_user_specified_namenormalization_25_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ѕ
ь
/__inference_sequential_77_layer_call_fn_1621106
normalization_25_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallnormalization_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621083o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
0
_user_specified_namenormalization_25_input:$ 

_output_shapes

::$ 

_output_shapes

:
ўH

 __inference__traced_save_1621680
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_308_kernel_read_readvariableop-
)savev2_dense_308_bias_read_readvariableop/
+savev2_dense_309_kernel_read_readvariableop-
)savev2_dense_309_bias_read_readvariableop/
+savev2_dense_310_kernel_read_readvariableop-
)savev2_dense_310_bias_read_readvariableop/
+savev2_dense_311_kernel_read_readvariableop-
)savev2_dense_311_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_2_read_readvariableop6
2savev2_adam_dense_308_kernel_m_read_readvariableop4
0savev2_adam_dense_308_bias_m_read_readvariableop6
2savev2_adam_dense_309_kernel_m_read_readvariableop4
0savev2_adam_dense_309_bias_m_read_readvariableop6
2savev2_adam_dense_310_kernel_m_read_readvariableop4
0savev2_adam_dense_310_bias_m_read_readvariableop6
2savev2_adam_dense_311_kernel_m_read_readvariableop4
0savev2_adam_dense_311_bias_m_read_readvariableop6
2savev2_adam_dense_308_kernel_v_read_readvariableop4
0savev2_adam_dense_308_bias_v_read_readvariableop6
2savev2_adam_dense_309_kernel_v_read_readvariableop4
0savev2_adam_dense_309_bias_v_read_readvariableop6
2savev2_adam_dense_310_kernel_v_read_readvariableop4
0savev2_adam_dense_310_bias_v_read_readvariableop6
2savev2_adam_dense_311_kernel_v_read_readvariableop4
0savev2_adam_dense_311_bias_v_read_readvariableop
savev2_const_2

identity_1ЂMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ђ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*Ы
valueСBО$B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЕ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ј
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_308_kernel_read_readvariableop)savev2_dense_308_bias_read_readvariableop+savev2_dense_309_kernel_read_readvariableop)savev2_dense_309_bias_read_readvariableop+savev2_dense_310_kernel_read_readvariableop)savev2_dense_310_bias_read_readvariableop+savev2_dense_311_kernel_read_readvariableop)savev2_dense_311_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_dense_308_kernel_m_read_readvariableop0savev2_adam_dense_308_bias_m_read_readvariableop2savev2_adam_dense_309_kernel_m_read_readvariableop0savev2_adam_dense_309_bias_m_read_readvariableop2savev2_adam_dense_310_kernel_m_read_readvariableop0savev2_adam_dense_310_bias_m_read_readvariableop2savev2_adam_dense_311_kernel_m_read_readvariableop0savev2_adam_dense_311_bias_m_read_readvariableop2savev2_adam_dense_308_kernel_v_read_readvariableop0savev2_adam_dense_308_bias_v_read_readvariableop2savev2_adam_dense_309_kernel_v_read_readvariableop0savev2_adam_dense_309_bias_v_read_readvariableop2savev2_adam_dense_310_kernel_v_read_readvariableop0savev2_adam_dense_310_bias_v_read_readvariableop2savev2_adam_dense_311_kernel_v_read_readvariableop0savev2_adam_dense_311_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$		
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ї
_input_shapesх
т: ::: ::::::::: : : : : : : : ::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 	

_output_shapes
::$
 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

:: #

_output_shapes
::$

_output_shapes
: 
Ы5
ю
"__inference__wrapped_model_1621001
normalization_25_input(
$sequential_77_normalization_25_sub_y)
%sequential_77_normalization_25_sqrt_xH
6sequential_77_dense_308_matmul_readvariableop_resource:E
7sequential_77_dense_308_biasadd_readvariableop_resource:H
6sequential_77_dense_309_matmul_readvariableop_resource:E
7sequential_77_dense_309_biasadd_readvariableop_resource:H
6sequential_77_dense_310_matmul_readvariableop_resource:E
7sequential_77_dense_310_biasadd_readvariableop_resource:H
6sequential_77_dense_311_matmul_readvariableop_resource:E
7sequential_77_dense_311_biasadd_readvariableop_resource:
identityЂ.sequential_77/dense_308/BiasAdd/ReadVariableOpЂ-sequential_77/dense_308/MatMul/ReadVariableOpЂ.sequential_77/dense_309/BiasAdd/ReadVariableOpЂ-sequential_77/dense_309/MatMul/ReadVariableOpЂ.sequential_77/dense_310/BiasAdd/ReadVariableOpЂ-sequential_77/dense_310/MatMul/ReadVariableOpЂ.sequential_77/dense_311/BiasAdd/ReadVariableOpЂ-sequential_77/dense_311/MatMul/ReadVariableOp
"sequential_77/normalization_25/subSubnormalization_25_input$sequential_77_normalization_25_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ{
#sequential_77/normalization_25/SqrtSqrt%sequential_77_normalization_25_sqrt_x*
T0*
_output_shapes

:m
(sequential_77/normalization_25/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3Ж
&sequential_77/normalization_25/MaximumMaximum'sequential_77/normalization_25/Sqrt:y:01sequential_77/normalization_25/Maximum/y:output:0*
T0*
_output_shapes

:З
&sequential_77/normalization_25/truedivRealDiv&sequential_77/normalization_25/sub:z:0*sequential_77/normalization_25/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
-sequential_77/dense_308/MatMul/ReadVariableOpReadVariableOp6sequential_77_dense_308_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Н
sequential_77/dense_308/MatMulMatMul*sequential_77/normalization_25/truediv:z:05sequential_77/dense_308/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
.sequential_77/dense_308/BiasAdd/ReadVariableOpReadVariableOp7sequential_77_dense_308_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
sequential_77/dense_308/BiasAddBiasAdd(sequential_77/dense_308/MatMul:product:06sequential_77/dense_308/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
sequential_77/dense_308/ReluRelu(sequential_77/dense_308/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
-sequential_77/dense_309/MatMul/ReadVariableOpReadVariableOp6sequential_77_dense_309_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Н
sequential_77/dense_309/MatMulMatMul*sequential_77/dense_308/Relu:activations:05sequential_77/dense_309/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
.sequential_77/dense_309/BiasAdd/ReadVariableOpReadVariableOp7sequential_77_dense_309_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
sequential_77/dense_309/BiasAddBiasAdd(sequential_77/dense_309/MatMul:product:06sequential_77/dense_309/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
sequential_77/dense_309/ReluRelu(sequential_77/dense_309/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
-sequential_77/dense_310/MatMul/ReadVariableOpReadVariableOp6sequential_77_dense_310_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Н
sequential_77/dense_310/MatMulMatMul*sequential_77/dense_309/Relu:activations:05sequential_77/dense_310/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
.sequential_77/dense_310/BiasAdd/ReadVariableOpReadVariableOp7sequential_77_dense_310_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
sequential_77/dense_310/BiasAddBiasAdd(sequential_77/dense_310/MatMul:product:06sequential_77/dense_310/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
sequential_77/dense_310/ReluRelu(sequential_77/dense_310/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
-sequential_77/dense_311/MatMul/ReadVariableOpReadVariableOp6sequential_77_dense_311_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Н
sequential_77/dense_311/MatMulMatMul*sequential_77/dense_310/Relu:activations:05sequential_77/dense_311/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЂ
.sequential_77/dense_311/BiasAdd/ReadVariableOpReadVariableOp7sequential_77_dense_311_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
sequential_77/dense_311/BiasAddBiasAdd(sequential_77/dense_311/MatMul:product:06sequential_77/dense_311/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџw
IdentityIdentity(sequential_77/dense_311/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЪ
NoOpNoOp/^sequential_77/dense_308/BiasAdd/ReadVariableOp.^sequential_77/dense_308/MatMul/ReadVariableOp/^sequential_77/dense_309/BiasAdd/ReadVariableOp.^sequential_77/dense_309/MatMul/ReadVariableOp/^sequential_77/dense_310/BiasAdd/ReadVariableOp.^sequential_77/dense_310/MatMul/ReadVariableOp/^sequential_77/dense_311/BiasAdd/ReadVariableOp.^sequential_77/dense_311/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2`
.sequential_77/dense_308/BiasAdd/ReadVariableOp.sequential_77/dense_308/BiasAdd/ReadVariableOp2^
-sequential_77/dense_308/MatMul/ReadVariableOp-sequential_77/dense_308/MatMul/ReadVariableOp2`
.sequential_77/dense_309/BiasAdd/ReadVariableOp.sequential_77/dense_309/BiasAdd/ReadVariableOp2^
-sequential_77/dense_309/MatMul/ReadVariableOp-sequential_77/dense_309/MatMul/ReadVariableOp2`
.sequential_77/dense_310/BiasAdd/ReadVariableOp.sequential_77/dense_310/BiasAdd/ReadVariableOp2^
-sequential_77/dense_310/MatMul/ReadVariableOp-sequential_77/dense_310/MatMul/ReadVariableOp2`
.sequential_77/dense_311/BiasAdd/ReadVariableOp.sequential_77/dense_311/BiasAdd/ReadVariableOp2^
-sequential_77/dense_311/MatMul/ReadVariableOp-sequential_77/dense_311/MatMul/ReadVariableOp:h d
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
0
_user_specified_namenormalization_25_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ѕ
ь
/__inference_sequential_77_layer_call_fn_1621252
normalization_25_input
unknown
	unknown_0
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
	unknown_8:
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallnormalization_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621204o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:h d
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
0
_user_specified_namenormalization_25_input:$ 

_output_shapes

::$ 

_output_shapes

:


ї
F__inference_dense_309_layer_call_and_return_conditional_losses_1621511

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ч*

J__inference_sequential_77_layer_call_and_return_conditional_losses_1621433

inputs
normalization_25_sub_y
normalization_25_sqrt_x:
(dense_308_matmul_readvariableop_resource:7
)dense_308_biasadd_readvariableop_resource::
(dense_309_matmul_readvariableop_resource:7
)dense_309_biasadd_readvariableop_resource::
(dense_310_matmul_readvariableop_resource:7
)dense_310_biasadd_readvariableop_resource::
(dense_311_matmul_readvariableop_resource:7
)dense_311_biasadd_readvariableop_resource:
identityЂ dense_308/BiasAdd/ReadVariableOpЂdense_308/MatMul/ReadVariableOpЂ dense_309/BiasAdd/ReadVariableOpЂdense_309/MatMul/ReadVariableOpЂ dense_310/BiasAdd/ReadVariableOpЂdense_310/MatMul/ReadVariableOpЂ dense_311/BiasAdd/ReadVariableOpЂdense_311/MatMul/ReadVariableOpm
normalization_25/subSubinputsnormalization_25_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_25/SqrtSqrtnormalization_25_sqrt_x*
T0*
_output_shapes

:_
normalization_25/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_25/MaximumMaximumnormalization_25/Sqrt:y:0#normalization_25/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_25/truedivRealDivnormalization_25/sub:z:0normalization_25/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_308/MatMul/ReadVariableOpReadVariableOp(dense_308_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_308/MatMulMatMulnormalization_25/truediv:z:0'dense_308/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_308/BiasAdd/ReadVariableOpReadVariableOp)dense_308_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_308/BiasAddBiasAdddense_308/MatMul:product:0(dense_308/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_308/ReluReludense_308/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_309/MatMul/ReadVariableOpReadVariableOp(dense_309_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_309/MatMulMatMuldense_308/Relu:activations:0'dense_309/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_309/BiasAdd/ReadVariableOpReadVariableOp)dense_309_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_309/BiasAddBiasAdddense_309/MatMul:product:0(dense_309/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_309/ReluReludense_309/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_310/MatMul/ReadVariableOpReadVariableOp(dense_310_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_310/MatMulMatMuldense_309/Relu:activations:0'dense_310/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_310/BiasAdd/ReadVariableOpReadVariableOp)dense_310_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_310/BiasAddBiasAdddense_310/MatMul:product:0(dense_310/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_310/ReluReludense_310/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_311/MatMul/ReadVariableOpReadVariableOp(dense_311_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_311/MatMulMatMuldense_310/Relu:activations:0'dense_311/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_311/BiasAdd/ReadVariableOpReadVariableOp)dense_311_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_311/BiasAddBiasAdddense_311/MatMul:product:0(dense_311/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџi
IdentityIdentitydense_311/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџк
NoOpNoOp!^dense_308/BiasAdd/ReadVariableOp ^dense_308/MatMul/ReadVariableOp!^dense_309/BiasAdd/ReadVariableOp ^dense_309/MatMul/ReadVariableOp!^dense_310/BiasAdd/ReadVariableOp ^dense_310/MatMul/ReadVariableOp!^dense_311/BiasAdd/ReadVariableOp ^dense_311/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2D
 dense_308/BiasAdd/ReadVariableOp dense_308/BiasAdd/ReadVariableOp2B
dense_308/MatMul/ReadVariableOpdense_308/MatMul/ReadVariableOp2D
 dense_309/BiasAdd/ReadVariableOp dense_309/BiasAdd/ReadVariableOp2B
dense_309/MatMul/ReadVariableOpdense_309/MatMul/ReadVariableOp2D
 dense_310/BiasAdd/ReadVariableOp dense_310/BiasAdd/ReadVariableOp2B
dense_310/MatMul/ReadVariableOpdense_310/MatMul/ReadVariableOp2D
 dense_311/BiasAdd/ReadVariableOp dense_311/BiasAdd/ReadVariableOp2B
dense_311/MatMul/ReadVariableOpdense_311/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*г
serving_defaultП
b
normalization_25_inputH
(serving_default_normalization_25_input:0џџџџџџџџџџџџџџџџџџ=
	dense_3110
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:Эf
Љ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
d__call__
*e&call_and_return_all_conditional_losses
f_default_save_signature"
_tf_keras_sequential
г

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean

adapt_mean
variance
adapt_variance
	count
	keras_api
g_adapt_function"
_tf_keras_layer
Л

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
Л

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
а
,iter

-beta_1

.beta_2
	/decaymTmUmVmW mX!mY&mZ'm[v\v]v^v_ v`!va&vb'vc"
	optimizer
n
0
1
2
3
4
5
6
 7
!8
&9
'10"
trackable_list_wrapper
X
0
1
2
3
 4
!5
&6
'7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
	regularization_losses
d__call__
f_default_save_signature
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
,
pserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
": 2dense_308/kernel
:2dense_308/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
": 2dense_309/kernel
:2dense_309/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
trainable_variables
regularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
": 2dense_310/kernel
:2dense_310/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
"	variables
#trainable_variables
$regularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
": 2dense_311/kernel
:2dense_311/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics
(	variables
)trainable_variables
*regularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
5
0
1
2"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	Ktotal
	Lcount
M	variables
N	keras_api"
_tf_keras_metric
^
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
K0
L1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
':%2Adam/dense_308/kernel/m
!:2Adam/dense_308/bias/m
':%2Adam/dense_309/kernel/m
!:2Adam/dense_309/bias/m
':%2Adam/dense_310/kernel/m
!:2Adam/dense_310/bias/m
':%2Adam/dense_311/kernel/m
!:2Adam/dense_311/bias/m
':%2Adam/dense_308/kernel/v
!:2Adam/dense_308/bias/v
':%2Adam/dense_309/kernel/v
!:2Adam/dense_309/bias/v
':%2Adam/dense_310/kernel/v
!:2Adam/dense_310/bias/v
':%2Adam/dense_311/kernel/v
!:2Adam/dense_311/bias/v
2
/__inference_sequential_77_layer_call_fn_1621106
/__inference_sequential_77_layer_call_fn_1621370
/__inference_sequential_77_layer_call_fn_1621395
/__inference_sequential_77_layer_call_fn_1621252Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621433
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621471
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621283
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621314Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
мBй
"__inference__wrapped_model_1621001normalization_25_input"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Р2Н
__inference_adapt_step_1616043
В
FullArgSpec
args

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_308_layer_call_fn_1621480Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_308_layer_call_and_return_conditional_losses_1621491Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_309_layer_call_fn_1621500Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_309_layer_call_and_return_conditional_losses_1621511Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_310_layer_call_fn_1621520Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_310_layer_call_and_return_conditional_losses_1621531Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_311_layer_call_fn_1621540Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_311_layer_call_and_return_conditional_losses_1621550Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
лBи
%__inference_signature_wrapper_1621345normalization_25_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
	J
Const
J	
Const_1Д
"__inference__wrapped_model_1621001
qr !&'HЂE
>Ђ;
96
normalization_25_inputџџџџџџџџџџџџџџџџџџ
Њ "5Њ2
0
	dense_311# 
	dense_311џџџџџџџџџp
__inference_adapt_step_1616043NCЂ@
9Ђ6
41Ђ
џџџџџџџџџIteratorSpec 
Њ "
 І
F__inference_dense_308_layer_call_and_return_conditional_losses_1621491\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_308_layer_call_fn_1621480O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
F__inference_dense_309_layer_call_and_return_conditional_losses_1621511\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_309_layer_call_fn_1621500O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
F__inference_dense_310_layer_call_and_return_conditional_losses_1621531\ !/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_310_layer_call_fn_1621520O !/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
F__inference_dense_311_layer_call_and_return_conditional_losses_1621550\&'/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_311_layer_call_fn_1621540O&'/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџд
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621283
qr !&'PЂM
FЂC
96
normalization_25_inputџџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 д
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621314
qr !&'PЂM
FЂC
96
normalization_25_inputџџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 У
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621433u
qr !&'@Ђ=
6Ђ3
)&
inputsџџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 У
J__inference_sequential_77_layer_call_and_return_conditional_losses_1621471u
qr !&'@Ђ=
6Ђ3
)&
inputsџџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ћ
/__inference_sequential_77_layer_call_fn_1621106x
qr !&'PЂM
FЂC
96
normalization_25_inputџџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџЋ
/__inference_sequential_77_layer_call_fn_1621252x
qr !&'PЂM
FЂC
96
normalization_25_inputџџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
/__inference_sequential_77_layer_call_fn_1621370h
qr !&'@Ђ=
6Ђ3
)&
inputsџџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
/__inference_sequential_77_layer_call_fn_1621395h
qr !&'@Ђ=
6Ђ3
)&
inputsџџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџб
%__inference_signature_wrapper_1621345Ї
qr !&'bЂ_
Ђ 
XЊU
S
normalization_25_input96
normalization_25_inputџџџџџџџџџџџџџџџџџџ"5Њ2
0
	dense_311# 
	dense_311џџџџџџџџџ