Я­
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
 "serve*2.7.02unknown8вД
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
dense_404/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_404/kernel
u
$dense_404/kernel/Read/ReadVariableOpReadVariableOpdense_404/kernel*
_output_shapes

:*
dtype0
t
dense_404/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_404/bias
m
"dense_404/bias/Read/ReadVariableOpReadVariableOpdense_404/bias*
_output_shapes
:*
dtype0
|
dense_405/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_405/kernel
u
$dense_405/kernel/Read/ReadVariableOpReadVariableOpdense_405/kernel*
_output_shapes

:*
dtype0
t
dense_405/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_405/bias
m
"dense_405/bias/Read/ReadVariableOpReadVariableOpdense_405/bias*
_output_shapes
:*
dtype0
|
dense_406/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_406/kernel
u
$dense_406/kernel/Read/ReadVariableOpReadVariableOpdense_406/kernel*
_output_shapes

:*
dtype0
t
dense_406/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_406/bias
m
"dense_406/bias/Read/ReadVariableOpReadVariableOpdense_406/bias*
_output_shapes
:*
dtype0
|
dense_407/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*!
shared_namedense_407/kernel
u
$dense_407/kernel/Read/ReadVariableOpReadVariableOpdense_407/kernel*
_output_shapes

:*
dtype0
t
dense_407/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_407/bias
m
"dense_407/bias/Read/ReadVariableOpReadVariableOpdense_407/bias*
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
Adam/dense_404/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_404/kernel/m

+Adam/dense_404/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_404/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_404/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_404/bias/m
{
)Adam/dense_404/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_404/bias/m*
_output_shapes
:*
dtype0

Adam/dense_405/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_405/kernel/m

+Adam/dense_405/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_405/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_405/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_405/bias/m
{
)Adam/dense_405/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_405/bias/m*
_output_shapes
:*
dtype0

Adam/dense_406/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_406/kernel/m

+Adam/dense_406/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_406/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_406/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_406/bias/m
{
)Adam/dense_406/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_406/bias/m*
_output_shapes
:*
dtype0

Adam/dense_407/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_407/kernel/m

+Adam/dense_407/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_407/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_407/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_407/bias/m
{
)Adam/dense_407/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_407/bias/m*
_output_shapes
:*
dtype0

Adam/dense_404/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_404/kernel/v

+Adam/dense_404/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_404/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_404/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_404/bias/v
{
)Adam/dense_404/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_404/bias/v*
_output_shapes
:*
dtype0

Adam/dense_405/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_405/kernel/v

+Adam/dense_405/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_405/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_405/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_405/bias/v
{
)Adam/dense_405/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_405/bias/v*
_output_shapes
:*
dtype0

Adam/dense_406/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_406/kernel/v

+Adam/dense_406/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_406/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_406/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_406/bias/v
{
)Adam/dense_406/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_406/bias/v*
_output_shapes
:*
dtype0

Adam/dense_407/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_nameAdam/dense_407/kernel/v

+Adam/dense_407/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_407/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_407/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/dense_407/bias/v
{
)Adam/dense_407/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_407/bias/v*
_output_shapes
:*
dtype0
І
ConstConst*
_output_shapes

:*
dtype0*i
value`B^"Pмі.D1D
,DSЅ.D<ЬЦGТ@>ъ6л8ии=h AЇQAии=Љі=Fњ==D§=
ь=*р=ь=фѕ=Мє=
Ј
Const_1Const*
_output_shapes

:*
dtype0*i
value`B^"PІKIЩМRI;мCIБЋJI ИPЗ<B(7n9EфEn9@W9Ћ09%H9ћ9ќу9$E9A92ц99

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
VARIABLE_VALUEdense_404/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_404/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_405/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_405/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_406/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_406/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEdense_407/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdense_407/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEAdam/dense_404/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_404/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_405/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_405/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_406/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_406/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_407/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_407/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_404/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_404/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_405/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_405/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_406/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_406/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_407/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense_407/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

&serving_default_normalization_33_inputPlaceholder*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
dtype0*%
shape:џџџџџџџџџџџџџџџџџџ
я
StatefulPartitionedCallStatefulPartitionedCall&serving_default_normalization_33_inputConstConst_1dense_404/kerneldense_404/biasdense_405/kerneldense_405/biasdense_406/kerneldense_406/biasdense_407/kerneldense_407/bias*
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
%__inference_signature_wrapper_2134639
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ђ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp$dense_404/kernel/Read/ReadVariableOp"dense_404/bias/Read/ReadVariableOp$dense_405/kernel/Read/ReadVariableOp"dense_405/bias/Read/ReadVariableOp$dense_406/kernel/Read/ReadVariableOp"dense_406/bias/Read/ReadVariableOp$dense_407/kernel/Read/ReadVariableOp"dense_407/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_2/Read/ReadVariableOp+Adam/dense_404/kernel/m/Read/ReadVariableOp)Adam/dense_404/bias/m/Read/ReadVariableOp+Adam/dense_405/kernel/m/Read/ReadVariableOp)Adam/dense_405/bias/m/Read/ReadVariableOp+Adam/dense_406/kernel/m/Read/ReadVariableOp)Adam/dense_406/bias/m/Read/ReadVariableOp+Adam/dense_407/kernel/m/Read/ReadVariableOp)Adam/dense_407/bias/m/Read/ReadVariableOp+Adam/dense_404/kernel/v/Read/ReadVariableOp)Adam/dense_404/bias/v/Read/ReadVariableOp+Adam/dense_405/kernel/v/Read/ReadVariableOp)Adam/dense_405/bias/v/Read/ReadVariableOp+Adam/dense_406/kernel/v/Read/ReadVariableOp)Adam/dense_406/bias/v/Read/ReadVariableOp+Adam/dense_407/kernel/v/Read/ReadVariableOp)Adam/dense_407/bias/v/Read/ReadVariableOpConst_2*0
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
 __inference__traced_save_2134974
Џ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_404/kerneldense_404/biasdense_405/kerneldense_405/biasdense_406/kerneldense_406/biasdense_407/kerneldense_407/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decaytotalcount_1total_1count_2Adam/dense_404/kernel/mAdam/dense_404/bias/mAdam/dense_405/kernel/mAdam/dense_405/bias/mAdam/dense_406/kernel/mAdam/dense_406/bias/mAdam/dense_407/kernel/mAdam/dense_407/bias/mAdam/dense_404/kernel/vAdam/dense_404/bias/vAdam/dense_405/kernel/vAdam/dense_405/bias/vAdam/dense_406/kernel/vAdam/dense_406/bias/vAdam/dense_407/kernel/vAdam/dense_407/bias/v*/
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
#__inference__traced_restore_2135089ЃЅ


ї
F__inference_dense_406_layer_call_and_return_conditional_losses_2134354

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
ќ
л
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134577
normalization_33_input
normalization_33_sub_y
normalization_33_sqrt_x#
dense_404_2134556:
dense_404_2134558:#
dense_405_2134561:
dense_405_2134563:#
dense_406_2134566:
dense_406_2134568:#
dense_407_2134571:
dense_407_2134573:
identityЂ!dense_404/StatefulPartitionedCallЂ!dense_405/StatefulPartitionedCallЂ!dense_406/StatefulPartitionedCallЂ!dense_407/StatefulPartitionedCall}
normalization_33/subSubnormalization_33_inputnormalization_33_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_33/SqrtSqrtnormalization_33_sqrt_x*
T0*
_output_shapes

:_
normalization_33/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_33/MaximumMaximumnormalization_33/Sqrt:y:0#normalization_33/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_33/truedivRealDivnormalization_33/sub:z:0normalization_33/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
!dense_404/StatefulPartitionedCallStatefulPartitionedCallnormalization_33/truediv:z:0dense_404_2134556dense_404_2134558*
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
F__inference_dense_404_layer_call_and_return_conditional_losses_2134320
!dense_405/StatefulPartitionedCallStatefulPartitionedCall*dense_404/StatefulPartitionedCall:output:0dense_405_2134561dense_405_2134563*
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
F__inference_dense_405_layer_call_and_return_conditional_losses_2134337
!dense_406/StatefulPartitionedCallStatefulPartitionedCall*dense_405/StatefulPartitionedCall:output:0dense_406_2134566dense_406_2134568*
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
F__inference_dense_406_layer_call_and_return_conditional_losses_2134354
!dense_407/StatefulPartitionedCallStatefulPartitionedCall*dense_406/StatefulPartitionedCall:output:0dense_407_2134571dense_407_2134573*
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
F__inference_dense_407_layer_call_and_return_conditional_losses_2134370y
IdentityIdentity*dense_407/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^dense_404/StatefulPartitionedCall"^dense_405/StatefulPartitionedCall"^dense_406/StatefulPartitionedCall"^dense_407/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2F
!dense_404/StatefulPartitionedCall!dense_404/StatefulPartitionedCall2F
!dense_405/StatefulPartitionedCall!dense_405/StatefulPartitionedCall2F
!dense_406/StatefulPartitionedCall!dense_406/StatefulPartitionedCall2F
!dense_407/StatefulPartitionedCall!dense_407/StatefulPartitionedCall:h d
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
0
_user_specified_namenormalization_33_input:$ 

_output_shapes

::$ 

_output_shapes

:


ї
F__inference_dense_406_layer_call_and_return_conditional_losses_2134825

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
Щ

+__inference_dense_404_layer_call_fn_2134774

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
F__inference_dense_404_layer_call_and_return_conditional_losses_2134320o
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


ї
F__inference_dense_404_layer_call_and_return_conditional_losses_2134785

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
Щ

+__inference_dense_406_layer_call_fn_2134814

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
F__inference_dense_406_layer_call_and_return_conditional_losses_2134354o
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
Џ6
	
"__inference__wrapped_model_2134295
normalization_33_input)
%sequential_101_normalization_33_sub_y*
&sequential_101_normalization_33_sqrt_xI
7sequential_101_dense_404_matmul_readvariableop_resource:F
8sequential_101_dense_404_biasadd_readvariableop_resource:I
7sequential_101_dense_405_matmul_readvariableop_resource:F
8sequential_101_dense_405_biasadd_readvariableop_resource:I
7sequential_101_dense_406_matmul_readvariableop_resource:F
8sequential_101_dense_406_biasadd_readvariableop_resource:I
7sequential_101_dense_407_matmul_readvariableop_resource:F
8sequential_101_dense_407_biasadd_readvariableop_resource:
identityЂ/sequential_101/dense_404/BiasAdd/ReadVariableOpЂ.sequential_101/dense_404/MatMul/ReadVariableOpЂ/sequential_101/dense_405/BiasAdd/ReadVariableOpЂ.sequential_101/dense_405/MatMul/ReadVariableOpЂ/sequential_101/dense_406/BiasAdd/ReadVariableOpЂ.sequential_101/dense_406/MatMul/ReadVariableOpЂ/sequential_101/dense_407/BiasAdd/ReadVariableOpЂ.sequential_101/dense_407/MatMul/ReadVariableOp
#sequential_101/normalization_33/subSubnormalization_33_input%sequential_101_normalization_33_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ}
$sequential_101/normalization_33/SqrtSqrt&sequential_101_normalization_33_sqrt_x*
T0*
_output_shapes

:n
)sequential_101/normalization_33/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3Й
'sequential_101/normalization_33/MaximumMaximum(sequential_101/normalization_33/Sqrt:y:02sequential_101/normalization_33/Maximum/y:output:0*
T0*
_output_shapes

:К
'sequential_101/normalization_33/truedivRealDiv'sequential_101/normalization_33/sub:z:0+sequential_101/normalization_33/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџІ
.sequential_101/dense_404/MatMul/ReadVariableOpReadVariableOp7sequential_101_dense_404_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Р
sequential_101/dense_404/MatMulMatMul+sequential_101/normalization_33/truediv:z:06sequential_101/dense_404/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
/sequential_101/dense_404/BiasAdd/ReadVariableOpReadVariableOp8sequential_101_dense_404_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
 sequential_101/dense_404/BiasAddBiasAdd)sequential_101/dense_404/MatMul:product:07sequential_101/dense_404/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
sequential_101/dense_404/ReluRelu)sequential_101/dense_404/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџІ
.sequential_101/dense_405/MatMul/ReadVariableOpReadVariableOp7sequential_101_dense_405_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Р
sequential_101/dense_405/MatMulMatMul+sequential_101/dense_404/Relu:activations:06sequential_101/dense_405/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
/sequential_101/dense_405/BiasAdd/ReadVariableOpReadVariableOp8sequential_101_dense_405_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
 sequential_101/dense_405/BiasAddBiasAdd)sequential_101/dense_405/MatMul:product:07sequential_101/dense_405/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
sequential_101/dense_405/ReluRelu)sequential_101/dense_405/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџІ
.sequential_101/dense_406/MatMul/ReadVariableOpReadVariableOp7sequential_101_dense_406_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Р
sequential_101/dense_406/MatMulMatMul+sequential_101/dense_405/Relu:activations:06sequential_101/dense_406/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
/sequential_101/dense_406/BiasAdd/ReadVariableOpReadVariableOp8sequential_101_dense_406_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
 sequential_101/dense_406/BiasAddBiasAdd)sequential_101/dense_406/MatMul:product:07sequential_101/dense_406/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
sequential_101/dense_406/ReluRelu)sequential_101/dense_406/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџІ
.sequential_101/dense_407/MatMul/ReadVariableOpReadVariableOp7sequential_101_dense_407_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Р
sequential_101/dense_407/MatMulMatMul+sequential_101/dense_406/Relu:activations:06sequential_101/dense_407/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџЄ
/sequential_101/dense_407/BiasAdd/ReadVariableOpReadVariableOp8sequential_101_dense_407_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
 sequential_101/dense_407/BiasAddBiasAdd)sequential_101/dense_407/MatMul:product:07sequential_101/dense_407/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџx
IdentityIdentity)sequential_101/dense_407/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџв
NoOpNoOp0^sequential_101/dense_404/BiasAdd/ReadVariableOp/^sequential_101/dense_404/MatMul/ReadVariableOp0^sequential_101/dense_405/BiasAdd/ReadVariableOp/^sequential_101/dense_405/MatMul/ReadVariableOp0^sequential_101/dense_406/BiasAdd/ReadVariableOp/^sequential_101/dense_406/MatMul/ReadVariableOp0^sequential_101/dense_407/BiasAdd/ReadVariableOp/^sequential_101/dense_407/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2b
/sequential_101/dense_404/BiasAdd/ReadVariableOp/sequential_101/dense_404/BiasAdd/ReadVariableOp2`
.sequential_101/dense_404/MatMul/ReadVariableOp.sequential_101/dense_404/MatMul/ReadVariableOp2b
/sequential_101/dense_405/BiasAdd/ReadVariableOp/sequential_101/dense_405/BiasAdd/ReadVariableOp2`
.sequential_101/dense_405/MatMul/ReadVariableOp.sequential_101/dense_405/MatMul/ReadVariableOp2b
/sequential_101/dense_406/BiasAdd/ReadVariableOp/sequential_101/dense_406/BiasAdd/ReadVariableOp2`
.sequential_101/dense_406/MatMul/ReadVariableOp.sequential_101/dense_406/MatMul/ReadVariableOp2b
/sequential_101/dense_407/BiasAdd/ReadVariableOp/sequential_101/dense_407/BiasAdd/ReadVariableOp2`
.sequential_101/dense_407/MatMul/ReadVariableOp.sequential_101/dense_407/MatMul/ReadVariableOp:h d
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
0
_user_specified_namenormalization_33_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ь
Ы
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134377

inputs
normalization_33_sub_y
normalization_33_sqrt_x#
dense_404_2134321:
dense_404_2134323:#
dense_405_2134338:
dense_405_2134340:#
dense_406_2134355:
dense_406_2134357:#
dense_407_2134371:
dense_407_2134373:
identityЂ!dense_404/StatefulPartitionedCallЂ!dense_405/StatefulPartitionedCallЂ!dense_406/StatefulPartitionedCallЂ!dense_407/StatefulPartitionedCallm
normalization_33/subSubinputsnormalization_33_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_33/SqrtSqrtnormalization_33_sqrt_x*
T0*
_output_shapes

:_
normalization_33/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_33/MaximumMaximumnormalization_33/Sqrt:y:0#normalization_33/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_33/truedivRealDivnormalization_33/sub:z:0normalization_33/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
!dense_404/StatefulPartitionedCallStatefulPartitionedCallnormalization_33/truediv:z:0dense_404_2134321dense_404_2134323*
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
F__inference_dense_404_layer_call_and_return_conditional_losses_2134320
!dense_405/StatefulPartitionedCallStatefulPartitionedCall*dense_404/StatefulPartitionedCall:output:0dense_405_2134338dense_405_2134340*
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
F__inference_dense_405_layer_call_and_return_conditional_losses_2134337
!dense_406/StatefulPartitionedCallStatefulPartitionedCall*dense_405/StatefulPartitionedCall:output:0dense_406_2134355dense_406_2134357*
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
F__inference_dense_406_layer_call_and_return_conditional_losses_2134354
!dense_407/StatefulPartitionedCallStatefulPartitionedCall*dense_406/StatefulPartitionedCall:output:0dense_407_2134371dense_407_2134373*
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
F__inference_dense_407_layer_call_and_return_conditional_losses_2134370y
IdentityIdentity*dense_407/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^dense_404/StatefulPartitionedCall"^dense_405/StatefulPartitionedCall"^dense_406/StatefulPartitionedCall"^dense_407/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2F
!dense_404/StatefulPartitionedCall!dense_404/StatefulPartitionedCall2F
!dense_405/StatefulPartitionedCall!dense_405/StatefulPartitionedCall2F
!dense_406/StatefulPartitionedCall!dense_406/StatefulPartitionedCall2F
!dense_407/StatefulPartitionedCall!dense_407/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
ї

н
0__inference_sequential_101_layer_call_fn_2134689

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
identityЂStatefulPartitionedCallЩ
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
GPU2*0J 8 *T
fORM
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134498o
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
Ї
э
0__inference_sequential_101_layer_call_fn_2134400
normalization_33_input
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
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallnormalization_33_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
GPU2*0J 8 *T
fORM
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134377o
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
_user_specified_namenormalization_33_input:$ 

_output_shapes

::$ 

_output_shapes

:


ї
F__inference_dense_405_layer_call_and_return_conditional_losses_2134337

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
ї

н
0__inference_sequential_101_layer_call_fn_2134664

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
identityЂStatefulPartitionedCallЩ
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
GPU2*0J 8 *T
fORM
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134377o
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
Ь
Ы
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134498

inputs
normalization_33_sub_y
normalization_33_sqrt_x#
dense_404_2134477:
dense_404_2134479:#
dense_405_2134482:
dense_405_2134484:#
dense_406_2134487:
dense_406_2134489:#
dense_407_2134492:
dense_407_2134494:
identityЂ!dense_404/StatefulPartitionedCallЂ!dense_405/StatefulPartitionedCallЂ!dense_406/StatefulPartitionedCallЂ!dense_407/StatefulPartitionedCallm
normalization_33/subSubinputsnormalization_33_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_33/SqrtSqrtnormalization_33_sqrt_x*
T0*
_output_shapes

:_
normalization_33/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_33/MaximumMaximumnormalization_33/Sqrt:y:0#normalization_33/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_33/truedivRealDivnormalization_33/sub:z:0normalization_33/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
!dense_404/StatefulPartitionedCallStatefulPartitionedCallnormalization_33/truediv:z:0dense_404_2134477dense_404_2134479*
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
F__inference_dense_404_layer_call_and_return_conditional_losses_2134320
!dense_405/StatefulPartitionedCallStatefulPartitionedCall*dense_404/StatefulPartitionedCall:output:0dense_405_2134482dense_405_2134484*
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
F__inference_dense_405_layer_call_and_return_conditional_losses_2134337
!dense_406/StatefulPartitionedCallStatefulPartitionedCall*dense_405/StatefulPartitionedCall:output:0dense_406_2134487dense_406_2134489*
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
F__inference_dense_406_layer_call_and_return_conditional_losses_2134354
!dense_407/StatefulPartitionedCallStatefulPartitionedCall*dense_406/StatefulPartitionedCall:output:0dense_407_2134492dense_407_2134494*
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
F__inference_dense_407_layer_call_and_return_conditional_losses_2134370y
IdentityIdentity*dense_407/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^dense_404/StatefulPartitionedCall"^dense_405/StatefulPartitionedCall"^dense_406/StatefulPartitionedCall"^dense_407/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2F
!dense_404/StatefulPartitionedCall!dense_404/StatefulPartitionedCall2F
!dense_405/StatefulPartitionedCall!dense_405/StatefulPartitionedCall2F
!dense_406/StatefulPartitionedCall!dense_406/StatefulPartitionedCall2F
!dense_407/StatefulPartitionedCall!dense_407/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
ќ
л
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134608
normalization_33_input
normalization_33_sub_y
normalization_33_sqrt_x#
dense_404_2134587:
dense_404_2134589:#
dense_405_2134592:
dense_405_2134594:#
dense_406_2134597:
dense_406_2134599:#
dense_407_2134602:
dense_407_2134604:
identityЂ!dense_404/StatefulPartitionedCallЂ!dense_405/StatefulPartitionedCallЂ!dense_406/StatefulPartitionedCallЂ!dense_407/StatefulPartitionedCall}
normalization_33/subSubnormalization_33_inputnormalization_33_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_33/SqrtSqrtnormalization_33_sqrt_x*
T0*
_output_shapes

:_
normalization_33/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_33/MaximumMaximumnormalization_33/Sqrt:y:0#normalization_33/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_33/truedivRealDivnormalization_33/sub:z:0normalization_33/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
!dense_404/StatefulPartitionedCallStatefulPartitionedCallnormalization_33/truediv:z:0dense_404_2134587dense_404_2134589*
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
F__inference_dense_404_layer_call_and_return_conditional_losses_2134320
!dense_405/StatefulPartitionedCallStatefulPartitionedCall*dense_404/StatefulPartitionedCall:output:0dense_405_2134592dense_405_2134594*
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
F__inference_dense_405_layer_call_and_return_conditional_losses_2134337
!dense_406/StatefulPartitionedCallStatefulPartitionedCall*dense_405/StatefulPartitionedCall:output:0dense_406_2134597dense_406_2134599*
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
F__inference_dense_406_layer_call_and_return_conditional_losses_2134354
!dense_407/StatefulPartitionedCallStatefulPartitionedCall*dense_406/StatefulPartitionedCall:output:0dense_407_2134602dense_407_2134604*
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
F__inference_dense_407_layer_call_and_return_conditional_losses_2134370y
IdentityIdentity*dense_407/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџж
NoOpNoOp"^dense_404/StatefulPartitionedCall"^dense_405/StatefulPartitionedCall"^dense_406/StatefulPartitionedCall"^dense_407/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2F
!dense_404/StatefulPartitionedCall!dense_404/StatefulPartitionedCall2F
!dense_405/StatefulPartitionedCall!dense_405/StatefulPartitionedCall2F
!dense_406/StatefulPartitionedCall!dense_406/StatefulPartitionedCall2F
!dense_407/StatefulPartitionedCall!dense_407/StatefulPartitionedCall:h d
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
0
_user_specified_namenormalization_33_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ш*

K__inference_sequential_101_layer_call_and_return_conditional_losses_2134765

inputs
normalization_33_sub_y
normalization_33_sqrt_x:
(dense_404_matmul_readvariableop_resource:7
)dense_404_biasadd_readvariableop_resource::
(dense_405_matmul_readvariableop_resource:7
)dense_405_biasadd_readvariableop_resource::
(dense_406_matmul_readvariableop_resource:7
)dense_406_biasadd_readvariableop_resource::
(dense_407_matmul_readvariableop_resource:7
)dense_407_biasadd_readvariableop_resource:
identityЂ dense_404/BiasAdd/ReadVariableOpЂdense_404/MatMul/ReadVariableOpЂ dense_405/BiasAdd/ReadVariableOpЂdense_405/MatMul/ReadVariableOpЂ dense_406/BiasAdd/ReadVariableOpЂdense_406/MatMul/ReadVariableOpЂ dense_407/BiasAdd/ReadVariableOpЂdense_407/MatMul/ReadVariableOpm
normalization_33/subSubinputsnormalization_33_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_33/SqrtSqrtnormalization_33_sqrt_x*
T0*
_output_shapes

:_
normalization_33/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_33/MaximumMaximumnormalization_33/Sqrt:y:0#normalization_33/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_33/truedivRealDivnormalization_33/sub:z:0normalization_33/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_404/MatMul/ReadVariableOpReadVariableOp(dense_404_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_404/MatMulMatMulnormalization_33/truediv:z:0'dense_404/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_404/BiasAdd/ReadVariableOpReadVariableOp)dense_404_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_404/BiasAddBiasAdddense_404/MatMul:product:0(dense_404/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_404/ReluReludense_404/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_405/MatMul/ReadVariableOpReadVariableOp(dense_405_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_405/MatMulMatMuldense_404/Relu:activations:0'dense_405/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_405/BiasAdd/ReadVariableOpReadVariableOp)dense_405_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_405/BiasAddBiasAdddense_405/MatMul:product:0(dense_405/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_405/ReluReludense_405/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_406/MatMul/ReadVariableOpReadVariableOp(dense_406_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_406/MatMulMatMuldense_405/Relu:activations:0'dense_406/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_406/BiasAdd/ReadVariableOpReadVariableOp)dense_406_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_406/BiasAddBiasAdddense_406/MatMul:product:0(dense_406/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_406/ReluReludense_406/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_407/MatMul/ReadVariableOpReadVariableOp(dense_407_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_407/MatMulMatMuldense_406/Relu:activations:0'dense_407/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_407/BiasAdd/ReadVariableOpReadVariableOp)dense_407_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_407/BiasAddBiasAdddense_407/MatMul:product:0(dense_407/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџi
IdentityIdentitydense_407/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџк
NoOpNoOp!^dense_404/BiasAdd/ReadVariableOp ^dense_404/MatMul/ReadVariableOp!^dense_405/BiasAdd/ReadVariableOp ^dense_405/MatMul/ReadVariableOp!^dense_406/BiasAdd/ReadVariableOp ^dense_406/MatMul/ReadVariableOp!^dense_407/BiasAdd/ReadVariableOp ^dense_407/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2D
 dense_404/BiasAdd/ReadVariableOp dense_404/BiasAdd/ReadVariableOp2B
dense_404/MatMul/ReadVariableOpdense_404/MatMul/ReadVariableOp2D
 dense_405/BiasAdd/ReadVariableOp dense_405/BiasAdd/ReadVariableOp2B
dense_405/MatMul/ReadVariableOpdense_405/MatMul/ReadVariableOp2D
 dense_406/BiasAdd/ReadVariableOp dense_406/BiasAdd/ReadVariableOp2B
dense_406/MatMul/ReadVariableOpdense_406/MatMul/ReadVariableOp2D
 dense_407/BiasAdd/ReadVariableOp dense_407/BiasAdd/ReadVariableOp2B
dense_407/MatMul/ReadVariableOpdense_407/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:


ї
F__inference_dense_405_layer_call_and_return_conditional_losses_2134805

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
ўH

 __inference__traced_save_2134974
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	/
+savev2_dense_404_kernel_read_readvariableop-
)savev2_dense_404_bias_read_readvariableop/
+savev2_dense_405_kernel_read_readvariableop-
)savev2_dense_405_bias_read_readvariableop/
+savev2_dense_406_kernel_read_readvariableop-
)savev2_dense_406_bias_read_readvariableop/
+savev2_dense_407_kernel_read_readvariableop-
)savev2_dense_407_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_2_read_readvariableop6
2savev2_adam_dense_404_kernel_m_read_readvariableop4
0savev2_adam_dense_404_bias_m_read_readvariableop6
2savev2_adam_dense_405_kernel_m_read_readvariableop4
0savev2_adam_dense_405_bias_m_read_readvariableop6
2savev2_adam_dense_406_kernel_m_read_readvariableop4
0savev2_adam_dense_406_bias_m_read_readvariableop6
2savev2_adam_dense_407_kernel_m_read_readvariableop4
0savev2_adam_dense_407_bias_m_read_readvariableop6
2savev2_adam_dense_404_kernel_v_read_readvariableop4
0savev2_adam_dense_404_bias_v_read_readvariableop6
2savev2_adam_dense_405_kernel_v_read_readvariableop4
0savev2_adam_dense_405_bias_v_read_readvariableop6
2savev2_adam_dense_406_kernel_v_read_readvariableop4
0savev2_adam_dense_406_bias_v_read_readvariableop6
2savev2_adam_dense_407_kernel_v_read_readvariableop4
0savev2_adam_dense_407_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop+savev2_dense_404_kernel_read_readvariableop)savev2_dense_404_bias_read_readvariableop+savev2_dense_405_kernel_read_readvariableop)savev2_dense_405_bias_read_readvariableop+savev2_dense_406_kernel_read_readvariableop)savev2_dense_406_bias_read_readvariableop+savev2_dense_407_kernel_read_readvariableop)savev2_dense_407_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_2_read_readvariableop2savev2_adam_dense_404_kernel_m_read_readvariableop0savev2_adam_dense_404_bias_m_read_readvariableop2savev2_adam_dense_405_kernel_m_read_readvariableop0savev2_adam_dense_405_bias_m_read_readvariableop2savev2_adam_dense_406_kernel_m_read_readvariableop0savev2_adam_dense_406_bias_m_read_readvariableop2savev2_adam_dense_407_kernel_m_read_readvariableop0savev2_adam_dense_407_bias_m_read_readvariableop2savev2_adam_dense_404_kernel_v_read_readvariableop0savev2_adam_dense_404_bias_v_read_readvariableop2savev2_adam_dense_405_kernel_v_read_readvariableop0savev2_adam_dense_405_bias_v_read_readvariableop2savev2_adam_dense_406_kernel_v_read_readvariableop0savev2_adam_dense_406_bias_v_read_readvariableop2savev2_adam_dense_407_kernel_v_read_readvariableop0savev2_adam_dense_407_bias_v_read_readvariableopsavev2_const_2"/device:CPU:0*
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
Ї
э
0__inference_sequential_101_layer_call_fn_2134546
normalization_33_input
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
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallnormalization_33_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
GPU2*0J 8 *T
fORM
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134498o
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
_user_specified_namenormalization_33_input:$ 

_output_shapes

::$ 

_output_shapes

:
Ш*

K__inference_sequential_101_layer_call_and_return_conditional_losses_2134727

inputs
normalization_33_sub_y
normalization_33_sqrt_x:
(dense_404_matmul_readvariableop_resource:7
)dense_404_biasadd_readvariableop_resource::
(dense_405_matmul_readvariableop_resource:7
)dense_405_biasadd_readvariableop_resource::
(dense_406_matmul_readvariableop_resource:7
)dense_406_biasadd_readvariableop_resource::
(dense_407_matmul_readvariableop_resource:7
)dense_407_biasadd_readvariableop_resource:
identityЂ dense_404/BiasAdd/ReadVariableOpЂdense_404/MatMul/ReadVariableOpЂ dense_405/BiasAdd/ReadVariableOpЂdense_405/MatMul/ReadVariableOpЂ dense_406/BiasAdd/ReadVariableOpЂdense_406/MatMul/ReadVariableOpЂ dense_407/BiasAdd/ReadVariableOpЂdense_407/MatMul/ReadVariableOpm
normalization_33/subSubinputsnormalization_33_sub_y*
T0*'
_output_shapes
:џџџџџџџџџ_
normalization_33/SqrtSqrtnormalization_33_sqrt_x*
T0*
_output_shapes

:_
normalization_33/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *Пж3
normalization_33/MaximumMaximumnormalization_33/Sqrt:y:0#normalization_33/Maximum/y:output:0*
T0*
_output_shapes

:
normalization_33/truedivRealDivnormalization_33/sub:z:0normalization_33/Maximum:z:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_404/MatMul/ReadVariableOpReadVariableOp(dense_404_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_404/MatMulMatMulnormalization_33/truediv:z:0'dense_404/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_404/BiasAdd/ReadVariableOpReadVariableOp)dense_404_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_404/BiasAddBiasAdddense_404/MatMul:product:0(dense_404/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_404/ReluReludense_404/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_405/MatMul/ReadVariableOpReadVariableOp(dense_405_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_405/MatMulMatMuldense_404/Relu:activations:0'dense_405/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_405/BiasAdd/ReadVariableOpReadVariableOp)dense_405_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_405/BiasAddBiasAdddense_405/MatMul:product:0(dense_405/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_405/ReluReludense_405/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_406/MatMul/ReadVariableOpReadVariableOp(dense_406_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_406/MatMulMatMuldense_405/Relu:activations:0'dense_406/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_406/BiasAdd/ReadVariableOpReadVariableOp)dense_406_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_406/BiasAddBiasAdddense_406/MatMul:product:0(dense_406/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџd
dense_406/ReluReludense_406/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
dense_407/MatMul/ReadVariableOpReadVariableOp(dense_407_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_407/MatMulMatMuldense_406/Relu:activations:0'dense_407/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 dense_407/BiasAdd/ReadVariableOpReadVariableOp)dense_407_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_407/BiasAddBiasAdddense_407/MatMul:product:0(dense_407/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџi
IdentityIdentitydense_407/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџк
NoOpNoOp!^dense_404/BiasAdd/ReadVariableOp ^dense_404/MatMul/ReadVariableOp!^dense_405/BiasAdd/ReadVariableOp ^dense_405/MatMul/ReadVariableOp!^dense_406/BiasAdd/ReadVariableOp ^dense_406/MatMul/ReadVariableOp!^dense_407/BiasAdd/ReadVariableOp ^dense_407/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:џџџџџџџџџџџџџџџџџџ::: : : : : : : : 2D
 dense_404/BiasAdd/ReadVariableOp dense_404/BiasAdd/ReadVariableOp2B
dense_404/MatMul/ReadVariableOpdense_404/MatMul/ReadVariableOp2D
 dense_405/BiasAdd/ReadVariableOp dense_405/BiasAdd/ReadVariableOp2B
dense_405/MatMul/ReadVariableOpdense_405/MatMul/ReadVariableOp2D
 dense_406/BiasAdd/ReadVariableOp dense_406/BiasAdd/ReadVariableOp2B
dense_406/MatMul/ReadVariableOpdense_406/MatMul/ReadVariableOp2D
 dense_407/BiasAdd/ReadVariableOp dense_407/BiasAdd/ReadVariableOp2B
dense_407/MatMul/ReadVariableOpdense_407/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:$ 

_output_shapes

::$ 

_output_shapes

:
Щ	
ї
F__inference_dense_407_layer_call_and_return_conditional_losses_2134844

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
ѓ

т
%__inference_signature_wrapper_2134639
normalization_33_input
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
StatefulPartitionedCallStatefulPartitionedCallnormalization_33_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
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
"__inference__wrapped_model_2134295o
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
_user_specified_namenormalization_33_input:$ 

_output_shapes

::$ 

_output_shapes

:
З
и
#__inference__traced_restore_2135089
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 5
#assignvariableop_3_dense_404_kernel:/
!assignvariableop_4_dense_404_bias:5
#assignvariableop_5_dense_405_kernel:/
!assignvariableop_6_dense_405_bias:5
#assignvariableop_7_dense_406_kernel:/
!assignvariableop_8_dense_406_bias:5
#assignvariableop_9_dense_407_kernel:0
"assignvariableop_10_dense_407_bias:'
assignvariableop_11_adam_iter:	 )
assignvariableop_12_adam_beta_1: )
assignvariableop_13_adam_beta_2: (
assignvariableop_14_adam_decay: #
assignvariableop_15_total: %
assignvariableop_16_count_1: %
assignvariableop_17_total_1: %
assignvariableop_18_count_2: =
+assignvariableop_19_adam_dense_404_kernel_m:7
)assignvariableop_20_adam_dense_404_bias_m:=
+assignvariableop_21_adam_dense_405_kernel_m:7
)assignvariableop_22_adam_dense_405_bias_m:=
+assignvariableop_23_adam_dense_406_kernel_m:7
)assignvariableop_24_adam_dense_406_bias_m:=
+assignvariableop_25_adam_dense_407_kernel_m:7
)assignvariableop_26_adam_dense_407_bias_m:=
+assignvariableop_27_adam_dense_404_kernel_v:7
)assignvariableop_28_adam_dense_404_bias_v:=
+assignvariableop_29_adam_dense_405_kernel_v:7
)assignvariableop_30_adam_dense_405_bias_v:=
+assignvariableop_31_adam_dense_406_kernel_v:7
)assignvariableop_32_adam_dense_406_bias_v:=
+assignvariableop_33_adam_dense_407_kernel_v:7
)assignvariableop_34_adam_dense_407_bias_v:
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
AssignVariableOp_3AssignVariableOp#assignvariableop_3_dense_404_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_404_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp#assignvariableop_5_dense_405_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_405_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp#assignvariableop_7_dense_406_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_406_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp#assignvariableop_9_dense_407_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_407_biasIdentity_10:output:0"/device:CPU:0*
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
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_dense_404_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_dense_404_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_dense_405_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_405_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_dense_406_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_406_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_dense_407_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_407_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_dense_404_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_404_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_dense_405_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_405_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_dense_406_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_dense_406_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_dense_407_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_407_bias_vIdentity_34:output:0"/device:CPU:0*
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
К'
г
__inference_adapt_step_2127097
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
Щ	
ї
F__inference_dense_407_layer_call_and_return_conditional_losses_2134370

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
F__inference_dense_404_layer_call_and_return_conditional_losses_2134320

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
Щ

+__inference_dense_407_layer_call_fn_2134834

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
F__inference_dense_407_layer_call_and_return_conditional_losses_2134370o
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
Щ

+__inference_dense_405_layer_call_fn_2134794

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
F__inference_dense_405_layer_call_and_return_conditional_losses_2134337o
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
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*г
serving_defaultП
b
normalization_33_inputH
(serving_default_normalization_33_input:0џџџџџџџџџџџџџџџџџџ=
	dense_4070
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:нf
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
": 2dense_404/kernel
:2dense_404/bias
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
": 2dense_405/kernel
:2dense_405/bias
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
": 2dense_406/kernel
:2dense_406/bias
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
": 2dense_407/kernel
:2dense_407/bias
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
':%2Adam/dense_404/kernel/m
!:2Adam/dense_404/bias/m
':%2Adam/dense_405/kernel/m
!:2Adam/dense_405/bias/m
':%2Adam/dense_406/kernel/m
!:2Adam/dense_406/bias/m
':%2Adam/dense_407/kernel/m
!:2Adam/dense_407/bias/m
':%2Adam/dense_404/kernel/v
!:2Adam/dense_404/bias/v
':%2Adam/dense_405/kernel/v
!:2Adam/dense_405/bias/v
':%2Adam/dense_406/kernel/v
!:2Adam/dense_406/bias/v
':%2Adam/dense_407/kernel/v
!:2Adam/dense_407/bias/v
2
0__inference_sequential_101_layer_call_fn_2134400
0__inference_sequential_101_layer_call_fn_2134664
0__inference_sequential_101_layer_call_fn_2134689
0__inference_sequential_101_layer_call_fn_2134546Р
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
њ2ї
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134727
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134765
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134577
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134608Р
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
"__inference__wrapped_model_2134295normalization_33_input"
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
__inference_adapt_step_2127097
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
+__inference_dense_404_layer_call_fn_2134774Ђ
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
F__inference_dense_404_layer_call_and_return_conditional_losses_2134785Ђ
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
+__inference_dense_405_layer_call_fn_2134794Ђ
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
F__inference_dense_405_layer_call_and_return_conditional_losses_2134805Ђ
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
+__inference_dense_406_layer_call_fn_2134814Ђ
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
F__inference_dense_406_layer_call_and_return_conditional_losses_2134825Ђ
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
+__inference_dense_407_layer_call_fn_2134834Ђ
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
F__inference_dense_407_layer_call_and_return_conditional_losses_2134844Ђ
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
%__inference_signature_wrapper_2134639normalization_33_input"
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
"__inference__wrapped_model_2134295
qr !&'HЂE
>Ђ;
96
normalization_33_inputџџџџџџџџџџџџџџџџџџ
Њ "5Њ2
0
	dense_407# 
	dense_407џџџџџџџџџp
__inference_adapt_step_2127097NCЂ@
9Ђ6
41Ђ
џџџџџџџџџIteratorSpec 
Њ "
 І
F__inference_dense_404_layer_call_and_return_conditional_losses_2134785\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_404_layer_call_fn_2134774O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
F__inference_dense_405_layer_call_and_return_conditional_losses_2134805\/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_405_layer_call_fn_2134794O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
F__inference_dense_406_layer_call_and_return_conditional_losses_2134825\ !/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_406_layer_call_fn_2134814O !/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџІ
F__inference_dense_407_layer_call_and_return_conditional_losses_2134844\&'/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ
 ~
+__inference_dense_407_layer_call_fn_2134834O&'/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "џџџџџџџџџе
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134577
qr !&'PЂM
FЂC
96
normalization_33_inputџџџџџџџџџџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ
 е
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134608
qr !&'PЂM
FЂC
96
normalization_33_inputџџџџџџџџџџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ
 Ф
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134727u
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
 Ф
K__inference_sequential_101_layer_call_and_return_conditional_losses_2134765u
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
 Ќ
0__inference_sequential_101_layer_call_fn_2134400x
qr !&'PЂM
FЂC
96
normalization_33_inputџџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџЌ
0__inference_sequential_101_layer_call_fn_2134546x
qr !&'PЂM
FЂC
96
normalization_33_inputџџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
0__inference_sequential_101_layer_call_fn_2134664h
qr !&'@Ђ=
6Ђ3
)&
inputsџџџџџџџџџџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
0__inference_sequential_101_layer_call_fn_2134689h
qr !&'@Ђ=
6Ђ3
)&
inputsџџџџџџџџџџџџџџџџџџ
p

 
Њ "џџџџџџџџџб
%__inference_signature_wrapper_2134639Ї
qr !&'bЂ_
Ђ 
XЊU
S
normalization_33_input96
normalization_33_inputџџџџџџџџџџџџџџџџџџ"5Њ2
0
	dense_407# 
	dense_407џџџџџџџџџ