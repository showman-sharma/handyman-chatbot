ß³
Ý
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.12v2.9.0-18-gd8ce9f9c3018À¡

RMSprop/Output-Layer/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameRMSprop/Output-Layer/bias/rms

1RMSprop/Output-Layer/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/Output-Layer/bias/rms*
_output_shapes
:*
dtype0

RMSprop/Output-Layer/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *0
shared_name!RMSprop/Output-Layer/kernel/rms

3RMSprop/Output-Layer/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/Output-Layer/kernel/rms*
_output_shapes

: *
dtype0

RMSprop/Hidden-Layer-3/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!RMSprop/Hidden-Layer-3/bias/rms

3RMSprop/Hidden-Layer-3/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/Hidden-Layer-3/bias/rms*
_output_shapes
: *
dtype0

!RMSprop/Hidden-Layer-3/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *2
shared_name#!RMSprop/Hidden-Layer-3/kernel/rms

5RMSprop/Hidden-Layer-3/kernel/rms/Read/ReadVariableOpReadVariableOp!RMSprop/Hidden-Layer-3/kernel/rms*
_output_shapes

:  *
dtype0

RMSprop/Hidden-Layer-2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!RMSprop/Hidden-Layer-2/bias/rms

3RMSprop/Hidden-Layer-2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/Hidden-Layer-2/bias/rms*
_output_shapes
: *
dtype0

!RMSprop/Hidden-Layer-2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *2
shared_name#!RMSprop/Hidden-Layer-2/kernel/rms

5RMSprop/Hidden-Layer-2/kernel/rms/Read/ReadVariableOpReadVariableOp!RMSprop/Hidden-Layer-2/kernel/rms*
_output_shapes

:  *
dtype0

RMSprop/Hidden-Layer-1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!RMSprop/Hidden-Layer-1/bias/rms

3RMSprop/Hidden-Layer-1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/Hidden-Layer-1/bias/rms*
_output_shapes
: *
dtype0

!RMSprop/Hidden-Layer-1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:I *2
shared_name#!RMSprop/Hidden-Layer-1/kernel/rms

5RMSprop/Hidden-Layer-1/kernel/rms/Read/ReadVariableOpReadVariableOp!RMSprop/Hidden-Layer-1/kernel/rms*
_output_shapes

:I *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
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
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
z
Output-Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameOutput-Layer/bias
s
%Output-Layer/bias/Read/ReadVariableOpReadVariableOpOutput-Layer/bias*
_output_shapes
:*
dtype0

Output-Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameOutput-Layer/kernel
{
'Output-Layer/kernel/Read/ReadVariableOpReadVariableOpOutput-Layer/kernel*
_output_shapes

: *
dtype0
~
Hidden-Layer-3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameHidden-Layer-3/bias
w
'Hidden-Layer-3/bias/Read/ReadVariableOpReadVariableOpHidden-Layer-3/bias*
_output_shapes
: *
dtype0

Hidden-Layer-3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameHidden-Layer-3/kernel

)Hidden-Layer-3/kernel/Read/ReadVariableOpReadVariableOpHidden-Layer-3/kernel*
_output_shapes

:  *
dtype0
~
Hidden-Layer-2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameHidden-Layer-2/bias
w
'Hidden-Layer-2/bias/Read/ReadVariableOpReadVariableOpHidden-Layer-2/bias*
_output_shapes
: *
dtype0

Hidden-Layer-2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameHidden-Layer-2/kernel

)Hidden-Layer-2/kernel/Read/ReadVariableOpReadVariableOpHidden-Layer-2/kernel*
_output_shapes

:  *
dtype0
~
Hidden-Layer-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameHidden-Layer-1/bias
w
'Hidden-Layer-1/bias/Read/ReadVariableOpReadVariableOpHidden-Layer-1/bias*
_output_shapes
: *
dtype0

Hidden-Layer-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:I *&
shared_nameHidden-Layer-1/kernel

)Hidden-Layer-1/kernel/Read/ReadVariableOpReadVariableOpHidden-Layer-1/kernel*
_output_shapes

:I *
dtype0

NoOpNoOp
µ2
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ð1
valueæ1Bã1 BÜ1
è
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
¦
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias*
<
0
1
2
3
$4
%5
,6
-7*
<
0
1
2
3
$4
%5
,6
-7*
* 
°
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
3trace_0
4trace_1
5trace_2
6trace_3* 
6
7trace_0
8trace_1
9trace_2
:trace_3* 
* 

;iter
	<decay
=learning_rate
>momentum
?rho	rmsh	rmsi	rmsj	rmsk	$rmsl	%rmsm	,rmsn	-rmso*

@serving_default* 

0
1*

0
1*
* 

Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ftrace_0* 

Gtrace_0* 
e_
VARIABLE_VALUEHidden-Layer-1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEHidden-Layer-1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Mtrace_0* 

Ntrace_0* 
e_
VARIABLE_VALUEHidden-Layer-2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEHidden-Layer-2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 

Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Ttrace_0* 

Utrace_0* 
e_
VARIABLE_VALUEHidden-Layer-3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEHidden-Layer-3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

,0
-1*

,0
-1*
* 

Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*

[trace_0* 

\trace_0* 
c]
VARIABLE_VALUEOutput-Layer/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEOutput-Layer/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

]0
^1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
OI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
_	variables
`	keras_api
	atotal
	bcount*
H
c	variables
d	keras_api
	etotal
	fcount
g
_fn_kwargs*

a0
b1*

_	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

e0
f1*

c	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUE!RMSprop/Hidden-Layer-1/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/Hidden-Layer-1/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!RMSprop/Hidden-Layer-2/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/Hidden-Layer-2/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!RMSprop/Hidden-Layer-3/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/Hidden-Layer-3/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/Output-Layer/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUERMSprop/Output-Layer/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*

$serving_default_Hidden-Layer-1_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿI
ú
StatefulPartitionedCallStatefulPartitionedCall$serving_default_Hidden-Layer-1_inputHidden-Layer-1/kernelHidden-Layer-1/biasHidden-Layer-2/kernelHidden-Layer-2/biasHidden-Layer-3/kernelHidden-Layer-3/biasOutput-Layer/kernelOutput-Layer/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_19925
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ò

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)Hidden-Layer-1/kernel/Read/ReadVariableOp'Hidden-Layer-1/bias/Read/ReadVariableOp)Hidden-Layer-2/kernel/Read/ReadVariableOp'Hidden-Layer-2/bias/Read/ReadVariableOp)Hidden-Layer-3/kernel/Read/ReadVariableOp'Hidden-Layer-3/bias/Read/ReadVariableOp'Output-Layer/kernel/Read/ReadVariableOp%Output-Layer/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp5RMSprop/Hidden-Layer-1/kernel/rms/Read/ReadVariableOp3RMSprop/Hidden-Layer-1/bias/rms/Read/ReadVariableOp5RMSprop/Hidden-Layer-2/kernel/rms/Read/ReadVariableOp3RMSprop/Hidden-Layer-2/bias/rms/Read/ReadVariableOp5RMSprop/Hidden-Layer-3/kernel/rms/Read/ReadVariableOp3RMSprop/Hidden-Layer-3/bias/rms/Read/ReadVariableOp3RMSprop/Output-Layer/kernel/rms/Read/ReadVariableOp1RMSprop/Output-Layer/bias/rms/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_20209
Ù
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameHidden-Layer-1/kernelHidden-Layer-1/biasHidden-Layer-2/kernelHidden-Layer-2/biasHidden-Layer-3/kernelHidden-Layer-3/biasOutput-Layer/kernelOutput-Layer/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototal_1count_1totalcount!RMSprop/Hidden-Layer-1/kernel/rmsRMSprop/Hidden-Layer-1/bias/rms!RMSprop/Hidden-Layer-2/kernel/rmsRMSprop/Hidden-Layer-2/bias/rms!RMSprop/Hidden-Layer-3/kernel/rmsRMSprop/Hidden-Layer-3/bias/rmsRMSprop/Output-Layer/kernel/rmsRMSprop/Output-Layer/bias/rms*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_20294¦
«1
Í
 __inference__wrapped_model_19626
hidden_layer_1_inputL
:sequential_1_hidden_layer_1_matmul_readvariableop_resource:I I
;sequential_1_hidden_layer_1_biasadd_readvariableop_resource: L
:sequential_1_hidden_layer_2_matmul_readvariableop_resource:  I
;sequential_1_hidden_layer_2_biasadd_readvariableop_resource: L
:sequential_1_hidden_layer_3_matmul_readvariableop_resource:  I
;sequential_1_hidden_layer_3_biasadd_readvariableop_resource: J
8sequential_1_output_layer_matmul_readvariableop_resource: G
9sequential_1_output_layer_biasadd_readvariableop_resource:
identity¢2sequential_1/Hidden-Layer-1/BiasAdd/ReadVariableOp¢1sequential_1/Hidden-Layer-1/MatMul/ReadVariableOp¢2sequential_1/Hidden-Layer-2/BiasAdd/ReadVariableOp¢1sequential_1/Hidden-Layer-2/MatMul/ReadVariableOp¢2sequential_1/Hidden-Layer-3/BiasAdd/ReadVariableOp¢1sequential_1/Hidden-Layer-3/MatMul/ReadVariableOp¢0sequential_1/Output-Layer/BiasAdd/ReadVariableOp¢/sequential_1/Output-Layer/MatMul/ReadVariableOp¬
1sequential_1/Hidden-Layer-1/MatMul/ReadVariableOpReadVariableOp:sequential_1_hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:I *
dtype0¯
"sequential_1/Hidden-Layer-1/MatMulMatMulhidden_layer_1_input9sequential_1/Hidden-Layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
2sequential_1/Hidden-Layer-1/BiasAdd/ReadVariableOpReadVariableOp;sequential_1_hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ê
#sequential_1/Hidden-Layer-1/BiasAddBiasAdd,sequential_1/Hidden-Layer-1/MatMul:product:0:sequential_1/Hidden-Layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 sequential_1/Hidden-Layer-1/ReluRelu,sequential_1/Hidden-Layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
1sequential_1/Hidden-Layer-2/MatMul/ReadVariableOpReadVariableOp:sequential_1_hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0É
"sequential_1/Hidden-Layer-2/MatMulMatMul.sequential_1/Hidden-Layer-1/Relu:activations:09sequential_1/Hidden-Layer-2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
2sequential_1/Hidden-Layer-2/BiasAdd/ReadVariableOpReadVariableOp;sequential_1_hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ê
#sequential_1/Hidden-Layer-2/BiasAddBiasAdd,sequential_1/Hidden-Layer-2/MatMul:product:0:sequential_1/Hidden-Layer-2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 sequential_1/Hidden-Layer-2/ReluRelu,sequential_1/Hidden-Layer-2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¬
1sequential_1/Hidden-Layer-3/MatMul/ReadVariableOpReadVariableOp:sequential_1_hidden_layer_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0É
"sequential_1/Hidden-Layer-3/MatMulMatMul.sequential_1/Hidden-Layer-2/Relu:activations:09sequential_1/Hidden-Layer-3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ª
2sequential_1/Hidden-Layer-3/BiasAdd/ReadVariableOpReadVariableOp;sequential_1_hidden_layer_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Ê
#sequential_1/Hidden-Layer-3/BiasAddBiasAdd,sequential_1/Hidden-Layer-3/MatMul:product:0:sequential_1/Hidden-Layer-3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 sequential_1/Hidden-Layer-3/ReluRelu,sequential_1/Hidden-Layer-3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ ¨
/sequential_1/Output-Layer/MatMul/ReadVariableOpReadVariableOp8sequential_1_output_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0Å
 sequential_1/Output-Layer/MatMulMatMul.sequential_1/Hidden-Layer-3/Relu:activations:07sequential_1/Output-Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0sequential_1/Output-Layer/BiasAdd/ReadVariableOpReadVariableOp9sequential_1_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!sequential_1/Output-Layer/BiasAddBiasAdd*sequential_1/Output-Layer/MatMul:product:08sequential_1/Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!sequential_1/Output-Layer/SoftmaxSoftmax*sequential_1/Output-Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
IdentityIdentity+sequential_1/Output-Layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
NoOpNoOp3^sequential_1/Hidden-Layer-1/BiasAdd/ReadVariableOp2^sequential_1/Hidden-Layer-1/MatMul/ReadVariableOp3^sequential_1/Hidden-Layer-2/BiasAdd/ReadVariableOp2^sequential_1/Hidden-Layer-2/MatMul/ReadVariableOp3^sequential_1/Hidden-Layer-3/BiasAdd/ReadVariableOp2^sequential_1/Hidden-Layer-3/MatMul/ReadVariableOp1^sequential_1/Output-Layer/BiasAdd/ReadVariableOp0^sequential_1/Output-Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 2h
2sequential_1/Hidden-Layer-1/BiasAdd/ReadVariableOp2sequential_1/Hidden-Layer-1/BiasAdd/ReadVariableOp2f
1sequential_1/Hidden-Layer-1/MatMul/ReadVariableOp1sequential_1/Hidden-Layer-1/MatMul/ReadVariableOp2h
2sequential_1/Hidden-Layer-2/BiasAdd/ReadVariableOp2sequential_1/Hidden-Layer-2/BiasAdd/ReadVariableOp2f
1sequential_1/Hidden-Layer-2/MatMul/ReadVariableOp1sequential_1/Hidden-Layer-2/MatMul/ReadVariableOp2h
2sequential_1/Hidden-Layer-3/BiasAdd/ReadVariableOp2sequential_1/Hidden-Layer-3/BiasAdd/ReadVariableOp2f
1sequential_1/Hidden-Layer-3/MatMul/ReadVariableOp1sequential_1/Hidden-Layer-3/MatMul/ReadVariableOp2d
0sequential_1/Output-Layer/BiasAdd/ReadVariableOp0sequential_1/Output-Layer/BiasAdd/ReadVariableOp2b
/sequential_1/Output-Layer/MatMul/ReadVariableOp/sequential_1/Output-Layer/MatMul/ReadVariableOp:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
.
_user_specified_nameHidden-Layer-1_input
 

ú
I__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_20051

inputs0
matmul_readvariableop_resource:I -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:I *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿI: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
 
_user_specified_nameinputs
 

ú
I__inference_Hidden-Layer-3_layer_call_and_return_conditional_losses_19678

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ä	
»
,__inference_sequential_1_layer_call_fn_19946

inputs
unknown:I 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_19702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
 
_user_specified_nameinputs
 

ú
I__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_19661

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ó
Â
G__inference_sequential_1_layer_call_and_return_conditional_losses_19872
hidden_layer_1_input&
hidden_layer_1_19851:I "
hidden_layer_1_19853: &
hidden_layer_2_19856:  "
hidden_layer_2_19858: &
hidden_layer_3_19861:  "
hidden_layer_3_19863: $
output_layer_19866:  
output_layer_19868:
identity¢&Hidden-Layer-1/StatefulPartitionedCall¢&Hidden-Layer-2/StatefulPartitionedCall¢&Hidden-Layer-3/StatefulPartitionedCall¢$Output-Layer/StatefulPartitionedCall
&Hidden-Layer-1/StatefulPartitionedCallStatefulPartitionedCallhidden_layer_1_inputhidden_layer_1_19851hidden_layer_1_19853*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_19644®
&Hidden-Layer-2/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-1/StatefulPartitionedCall:output:0hidden_layer_2_19856hidden_layer_2_19858*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_19661®
&Hidden-Layer-3/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-2/StatefulPartitionedCall:output:0hidden_layer_3_19861hidden_layer_3_19863*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-3_layer_call_and_return_conditional_losses_19678¦
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-3/StatefulPartitionedCall:output:0output_layer_19866output_layer_19868*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Output-Layer_layer_call_and_return_conditional_losses_19695|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp'^Hidden-Layer-1/StatefulPartitionedCall'^Hidden-Layer-2/StatefulPartitionedCall'^Hidden-Layer-3/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 2P
&Hidden-Layer-1/StatefulPartitionedCall&Hidden-Layer-1/StatefulPartitionedCall2P
&Hidden-Layer-2/StatefulPartitionedCall&Hidden-Layer-2/StatefulPartitionedCall2P
&Hidden-Layer-3/StatefulPartitionedCall&Hidden-Layer-3/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
.
_user_specified_nameHidden-Layer-1_input
Ì

.__inference_Hidden-Layer-1_layer_call_fn_20040

inputs
unknown:I 
	unknown_0: 
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_19644o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿI: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
 
_user_specified_nameinputs
Ì

.__inference_Hidden-Layer-2_layer_call_fn_20060

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_19661o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬(

G__inference_sequential_1_layer_call_and_return_conditional_losses_19999

inputs?
-hidden_layer_1_matmul_readvariableop_resource:I <
.hidden_layer_1_biasadd_readvariableop_resource: ?
-hidden_layer_2_matmul_readvariableop_resource:  <
.hidden_layer_2_biasadd_readvariableop_resource: ?
-hidden_layer_3_matmul_readvariableop_resource:  <
.hidden_layer_3_biasadd_readvariableop_resource: =
+output_layer_matmul_readvariableop_resource: :
,output_layer_biasadd_readvariableop_resource:
identity¢%Hidden-Layer-1/BiasAdd/ReadVariableOp¢$Hidden-Layer-1/MatMul/ReadVariableOp¢%Hidden-Layer-2/BiasAdd/ReadVariableOp¢$Hidden-Layer-2/MatMul/ReadVariableOp¢%Hidden-Layer-3/BiasAdd/ReadVariableOp¢$Hidden-Layer-3/MatMul/ReadVariableOp¢#Output-Layer/BiasAdd/ReadVariableOp¢"Output-Layer/MatMul/ReadVariableOp
$Hidden-Layer-1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:I *
dtype0
Hidden-Layer-1/MatMulMatMulinputs,Hidden-Layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%Hidden-Layer-1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0£
Hidden-Layer-1/BiasAddBiasAddHidden-Layer-1/MatMul:product:0-Hidden-Layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
Hidden-Layer-1/ReluReluHidden-Layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$Hidden-Layer-2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0¢
Hidden-Layer-2/MatMulMatMul!Hidden-Layer-1/Relu:activations:0,Hidden-Layer-2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%Hidden-Layer-2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0£
Hidden-Layer-2/BiasAddBiasAddHidden-Layer-2/MatMul:product:0-Hidden-Layer-2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
Hidden-Layer-2/ReluReluHidden-Layer-2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$Hidden-Layer-3/MatMul/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0¢
Hidden-Layer-3/MatMulMatMul!Hidden-Layer-2/Relu:activations:0,Hidden-Layer-3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%Hidden-Layer-3/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0£
Hidden-Layer-3/BiasAddBiasAddHidden-Layer-3/MatMul:product:0-Hidden-Layer-3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
Hidden-Layer-3/ReluReluHidden-Layer-3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Output-Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
Output-Layer/MatMulMatMul!Hidden-Layer-3/Relu:activations:0*Output-Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Output-Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Output-Layer/BiasAddBiasAddOutput-Layer/MatMul:product:0+Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
Output-Layer/SoftmaxSoftmaxOutput-Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentityOutput-Layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
NoOpNoOp&^Hidden-Layer-1/BiasAdd/ReadVariableOp%^Hidden-Layer-1/MatMul/ReadVariableOp&^Hidden-Layer-2/BiasAdd/ReadVariableOp%^Hidden-Layer-2/MatMul/ReadVariableOp&^Hidden-Layer-3/BiasAdd/ReadVariableOp%^Hidden-Layer-3/MatMul/ReadVariableOp$^Output-Layer/BiasAdd/ReadVariableOp#^Output-Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 2N
%Hidden-Layer-1/BiasAdd/ReadVariableOp%Hidden-Layer-1/BiasAdd/ReadVariableOp2L
$Hidden-Layer-1/MatMul/ReadVariableOp$Hidden-Layer-1/MatMul/ReadVariableOp2N
%Hidden-Layer-2/BiasAdd/ReadVariableOp%Hidden-Layer-2/BiasAdd/ReadVariableOp2L
$Hidden-Layer-2/MatMul/ReadVariableOp$Hidden-Layer-2/MatMul/ReadVariableOp2N
%Hidden-Layer-3/BiasAdd/ReadVariableOp%Hidden-Layer-3/BiasAdd/ReadVariableOp2L
$Hidden-Layer-3/MatMul/ReadVariableOp$Hidden-Layer-3/MatMul/ReadVariableOp2J
#Output-Layer/BiasAdd/ReadVariableOp#Output-Layer/BiasAdd/ReadVariableOp2H
"Output-Layer/MatMul/ReadVariableOp"Output-Layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
 
_user_specified_nameinputs
f
Ï
!__inference__traced_restore_20294
file_prefix8
&assignvariableop_hidden_layer_1_kernel:I 4
&assignvariableop_1_hidden_layer_1_bias: :
(assignvariableop_2_hidden_layer_2_kernel:  4
&assignvariableop_3_hidden_layer_2_bias: :
(assignvariableop_4_hidden_layer_3_kernel:  4
&assignvariableop_5_hidden_layer_3_bias: 8
&assignvariableop_6_output_layer_kernel: 2
$assignvariableop_7_output_layer_bias:)
assignvariableop_8_rmsprop_iter:	 *
 assignvariableop_9_rmsprop_decay: 3
)assignvariableop_10_rmsprop_learning_rate: .
$assignvariableop_11_rmsprop_momentum: )
assignvariableop_12_rmsprop_rho: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: G
5assignvariableop_17_rmsprop_hidden_layer_1_kernel_rms:I A
3assignvariableop_18_rmsprop_hidden_layer_1_bias_rms: G
5assignvariableop_19_rmsprop_hidden_layer_2_kernel_rms:  A
3assignvariableop_20_rmsprop_hidden_layer_2_bias_rms: G
5assignvariableop_21_rmsprop_hidden_layer_3_kernel_rms:  A
3assignvariableop_22_rmsprop_hidden_layer_3_bias_rms: E
3assignvariableop_23_rmsprop_output_layer_kernel_rms: ?
1assignvariableop_24_rmsprop_output_layer_bias_rms:
identity_26¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9©
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ï
valueÅBÂB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B  
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp&assignvariableop_hidden_layer_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp&assignvariableop_1_hidden_layer_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp(assignvariableop_4_hidden_layer_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp&assignvariableop_5_hidden_layer_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp&assignvariableop_6_output_layer_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp$assignvariableop_7_output_layer_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_rmsprop_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp)assignvariableop_10_rmsprop_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp$assignvariableop_11_rmsprop_momentumIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_rmsprop_rhoIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
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
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_17AssignVariableOp5assignvariableop_17_rmsprop_hidden_layer_1_kernel_rmsIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_18AssignVariableOp3assignvariableop_18_rmsprop_hidden_layer_1_bias_rmsIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_19AssignVariableOp5assignvariableop_19_rmsprop_hidden_layer_2_kernel_rmsIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_20AssignVariableOp3assignvariableop_20_rmsprop_hidden_layer_2_bias_rmsIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_21AssignVariableOp5assignvariableop_21_rmsprop_hidden_layer_3_kernel_rmsIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_22AssignVariableOp3assignvariableop_22_rmsprop_hidden_layer_3_bias_rmsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_23AssignVariableOp3assignvariableop_23_rmsprop_output_layer_kernel_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¢
AssignVariableOp_24AssignVariableOp1assignvariableop_24_rmsprop_output_layer_bias_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 õ
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: â
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
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
î	
É
,__inference_sequential_1_layer_call_fn_19721
hidden_layer_1_input
unknown:I 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallhidden_layer_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_19702o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
.
_user_specified_nameHidden-Layer-1_input
É
´
G__inference_sequential_1_layer_call_and_return_conditional_losses_19808

inputs&
hidden_layer_1_19787:I "
hidden_layer_1_19789: &
hidden_layer_2_19792:  "
hidden_layer_2_19794: &
hidden_layer_3_19797:  "
hidden_layer_3_19799: $
output_layer_19802:  
output_layer_19804:
identity¢&Hidden-Layer-1/StatefulPartitionedCall¢&Hidden-Layer-2/StatefulPartitionedCall¢&Hidden-Layer-3/StatefulPartitionedCall¢$Output-Layer/StatefulPartitionedCall
&Hidden-Layer-1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_1_19787hidden_layer_1_19789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_19644®
&Hidden-Layer-2/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-1/StatefulPartitionedCall:output:0hidden_layer_2_19792hidden_layer_2_19794*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_19661®
&Hidden-Layer-3/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-2/StatefulPartitionedCall:output:0hidden_layer_3_19797hidden_layer_3_19799*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-3_layer_call_and_return_conditional_losses_19678¦
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-3/StatefulPartitionedCall:output:0output_layer_19802output_layer_19804*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Output-Layer_layer_call_and_return_conditional_losses_19695|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp'^Hidden-Layer-1/StatefulPartitionedCall'^Hidden-Layer-2/StatefulPartitionedCall'^Hidden-Layer-3/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 2P
&Hidden-Layer-1/StatefulPartitionedCall&Hidden-Layer-1/StatefulPartitionedCall2P
&Hidden-Layer-2/StatefulPartitionedCall&Hidden-Layer-2/StatefulPartitionedCall2P
&Hidden-Layer-3/StatefulPartitionedCall&Hidden-Layer-3/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
 
_user_specified_nameinputs
É
´
G__inference_sequential_1_layer_call_and_return_conditional_losses_19702

inputs&
hidden_layer_1_19645:I "
hidden_layer_1_19647: &
hidden_layer_2_19662:  "
hidden_layer_2_19664: &
hidden_layer_3_19679:  "
hidden_layer_3_19681: $
output_layer_19696:  
output_layer_19698:
identity¢&Hidden-Layer-1/StatefulPartitionedCall¢&Hidden-Layer-2/StatefulPartitionedCall¢&Hidden-Layer-3/StatefulPartitionedCall¢$Output-Layer/StatefulPartitionedCall
&Hidden-Layer-1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_1_19645hidden_layer_1_19647*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_19644®
&Hidden-Layer-2/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-1/StatefulPartitionedCall:output:0hidden_layer_2_19662hidden_layer_2_19664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_19661®
&Hidden-Layer-3/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-2/StatefulPartitionedCall:output:0hidden_layer_3_19679hidden_layer_3_19681*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-3_layer_call_and_return_conditional_losses_19678¦
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-3/StatefulPartitionedCall:output:0output_layer_19696output_layer_19698*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Output-Layer_layer_call_and_return_conditional_losses_19695|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp'^Hidden-Layer-1/StatefulPartitionedCall'^Hidden-Layer-2/StatefulPartitionedCall'^Hidden-Layer-3/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 2P
&Hidden-Layer-1/StatefulPartitionedCall&Hidden-Layer-1/StatefulPartitionedCall2P
&Hidden-Layer-2/StatefulPartitionedCall&Hidden-Layer-2/StatefulPartitionedCall2P
&Hidden-Layer-3/StatefulPartitionedCall&Hidden-Layer-3/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
 
_user_specified_nameinputs
 

ú
I__inference_Hidden-Layer-3_layer_call_and_return_conditional_losses_20091

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ä	
»
,__inference_sequential_1_layer_call_fn_19967

inputs
unknown:I 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_19808o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
 
_user_specified_nameinputs
ó
Â
G__inference_sequential_1_layer_call_and_return_conditional_losses_19896
hidden_layer_1_input&
hidden_layer_1_19875:I "
hidden_layer_1_19877: &
hidden_layer_2_19880:  "
hidden_layer_2_19882: &
hidden_layer_3_19885:  "
hidden_layer_3_19887: $
output_layer_19890:  
output_layer_19892:
identity¢&Hidden-Layer-1/StatefulPartitionedCall¢&Hidden-Layer-2/StatefulPartitionedCall¢&Hidden-Layer-3/StatefulPartitionedCall¢$Output-Layer/StatefulPartitionedCall
&Hidden-Layer-1/StatefulPartitionedCallStatefulPartitionedCallhidden_layer_1_inputhidden_layer_1_19875hidden_layer_1_19877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_19644®
&Hidden-Layer-2/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-1/StatefulPartitionedCall:output:0hidden_layer_2_19880hidden_layer_2_19882*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_19661®
&Hidden-Layer-3/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-2/StatefulPartitionedCall:output:0hidden_layer_3_19885hidden_layer_3_19887*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-3_layer_call_and_return_conditional_losses_19678¦
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-3/StatefulPartitionedCall:output:0output_layer_19890output_layer_19892*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Output-Layer_layer_call_and_return_conditional_losses_19695|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp'^Hidden-Layer-1/StatefulPartitionedCall'^Hidden-Layer-2/StatefulPartitionedCall'^Hidden-Layer-3/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 2P
&Hidden-Layer-1/StatefulPartitionedCall&Hidden-Layer-1/StatefulPartitionedCall2P
&Hidden-Layer-2/StatefulPartitionedCall&Hidden-Layer-2/StatefulPartitionedCall2P
&Hidden-Layer-3/StatefulPartitionedCall&Hidden-Layer-3/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
.
_user_specified_nameHidden-Layer-1_input
£

ø
G__inference_Output-Layer_layer_call_and_return_conditional_losses_20111

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
¬(

G__inference_sequential_1_layer_call_and_return_conditional_losses_20031

inputs?
-hidden_layer_1_matmul_readvariableop_resource:I <
.hidden_layer_1_biasadd_readvariableop_resource: ?
-hidden_layer_2_matmul_readvariableop_resource:  <
.hidden_layer_2_biasadd_readvariableop_resource: ?
-hidden_layer_3_matmul_readvariableop_resource:  <
.hidden_layer_3_biasadd_readvariableop_resource: =
+output_layer_matmul_readvariableop_resource: :
,output_layer_biasadd_readvariableop_resource:
identity¢%Hidden-Layer-1/BiasAdd/ReadVariableOp¢$Hidden-Layer-1/MatMul/ReadVariableOp¢%Hidden-Layer-2/BiasAdd/ReadVariableOp¢$Hidden-Layer-2/MatMul/ReadVariableOp¢%Hidden-Layer-3/BiasAdd/ReadVariableOp¢$Hidden-Layer-3/MatMul/ReadVariableOp¢#Output-Layer/BiasAdd/ReadVariableOp¢"Output-Layer/MatMul/ReadVariableOp
$Hidden-Layer-1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:I *
dtype0
Hidden-Layer-1/MatMulMatMulinputs,Hidden-Layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%Hidden-Layer-1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0£
Hidden-Layer-1/BiasAddBiasAddHidden-Layer-1/MatMul:product:0-Hidden-Layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
Hidden-Layer-1/ReluReluHidden-Layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$Hidden-Layer-2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0¢
Hidden-Layer-2/MatMulMatMul!Hidden-Layer-1/Relu:activations:0,Hidden-Layer-2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%Hidden-Layer-2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0£
Hidden-Layer-2/BiasAddBiasAddHidden-Layer-2/MatMul:product:0-Hidden-Layer-2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
Hidden-Layer-2/ReluReluHidden-Layer-2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
$Hidden-Layer-3/MatMul/ReadVariableOpReadVariableOp-hidden_layer_3_matmul_readvariableop_resource*
_output_shapes

:  *
dtype0¢
Hidden-Layer-3/MatMulMatMul!Hidden-Layer-2/Relu:activations:0,Hidden-Layer-3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
%Hidden-Layer-3/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0£
Hidden-Layer-3/BiasAddBiasAddHidden-Layer-3/MatMul:product:0-Hidden-Layer-3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ n
Hidden-Layer-3/ReluReluHidden-Layer-3/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
"Output-Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

: *
dtype0
Output-Layer/MatMulMatMul!Hidden-Layer-3/Relu:activations:0*Output-Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Output-Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Output-Layer/BiasAddBiasAddOutput-Layer/MatMul:product:0+Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
Output-Layer/SoftmaxSoftmaxOutput-Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentityOutput-Layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
NoOpNoOp&^Hidden-Layer-1/BiasAdd/ReadVariableOp%^Hidden-Layer-1/MatMul/ReadVariableOp&^Hidden-Layer-2/BiasAdd/ReadVariableOp%^Hidden-Layer-2/MatMul/ReadVariableOp&^Hidden-Layer-3/BiasAdd/ReadVariableOp%^Hidden-Layer-3/MatMul/ReadVariableOp$^Output-Layer/BiasAdd/ReadVariableOp#^Output-Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 2N
%Hidden-Layer-1/BiasAdd/ReadVariableOp%Hidden-Layer-1/BiasAdd/ReadVariableOp2L
$Hidden-Layer-1/MatMul/ReadVariableOp$Hidden-Layer-1/MatMul/ReadVariableOp2N
%Hidden-Layer-2/BiasAdd/ReadVariableOp%Hidden-Layer-2/BiasAdd/ReadVariableOp2L
$Hidden-Layer-2/MatMul/ReadVariableOp$Hidden-Layer-2/MatMul/ReadVariableOp2N
%Hidden-Layer-3/BiasAdd/ReadVariableOp%Hidden-Layer-3/BiasAdd/ReadVariableOp2L
$Hidden-Layer-3/MatMul/ReadVariableOp$Hidden-Layer-3/MatMul/ReadVariableOp2J
#Output-Layer/BiasAdd/ReadVariableOp#Output-Layer/BiasAdd/ReadVariableOp2H
"Output-Layer/MatMul/ReadVariableOp"Output-Layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
 
_user_specified_nameinputs
Ì

.__inference_Hidden-Layer-3_layer_call_fn_20080

inputs
unknown:  
	unknown_0: 
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_Hidden-Layer-3_layer_call_and_return_conditional_losses_19678o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
Ü9

__inference__traced_save_20209
file_prefix4
0savev2_hidden_layer_1_kernel_read_readvariableop2
.savev2_hidden_layer_1_bias_read_readvariableop4
0savev2_hidden_layer_2_kernel_read_readvariableop2
.savev2_hidden_layer_2_bias_read_readvariableop4
0savev2_hidden_layer_3_kernel_read_readvariableop2
.savev2_hidden_layer_3_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop@
<savev2_rmsprop_hidden_layer_1_kernel_rms_read_readvariableop>
:savev2_rmsprop_hidden_layer_1_bias_rms_read_readvariableop@
<savev2_rmsprop_hidden_layer_2_kernel_rms_read_readvariableop>
:savev2_rmsprop_hidden_layer_2_bias_rms_read_readvariableop@
<savev2_rmsprop_hidden_layer_3_kernel_rms_read_readvariableop>
:savev2_rmsprop_hidden_layer_3_bias_rms_read_readvariableop>
:savev2_rmsprop_output_layer_kernel_rms_read_readvariableop<
8savev2_rmsprop_output_layer_bias_rms_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
: ¦
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ï
valueÅBÂB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¡
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop0savev2_hidden_layer_2_kernel_read_readvariableop.savev2_hidden_layer_2_bias_read_readvariableop0savev2_hidden_layer_3_kernel_read_readvariableop.savev2_hidden_layer_3_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop<savev2_rmsprop_hidden_layer_1_kernel_rms_read_readvariableop:savev2_rmsprop_hidden_layer_1_bias_rms_read_readvariableop<savev2_rmsprop_hidden_layer_2_kernel_rms_read_readvariableop:savev2_rmsprop_hidden_layer_2_bias_rms_read_readvariableop<savev2_rmsprop_hidden_layer_3_kernel_rms_read_readvariableop:savev2_rmsprop_hidden_layer_3_bias_rms_read_readvariableop:savev2_rmsprop_output_layer_kernel_rms_read_readvariableop8savev2_rmsprop_output_layer_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	
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

identity_1Identity_1:output:0*«
_input_shapes
: :I : :  : :  : : :: : : : : : : : : :I : :  : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:I : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :
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
: :$ 

_output_shapes

:I : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: 
 

ú
I__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_19644

inputs0
matmul_readvariableop_resource:I -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:I *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿI: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
 
_user_specified_nameinputs
¾	
À
#__inference_signature_wrapper_19925
hidden_layer_1_input
unknown:I 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallhidden_layer_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_19626o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
.
_user_specified_nameHidden-Layer-1_input
£

ø
G__inference_Output-Layer_layer_call_and_return_conditional_losses_19695

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
 

ú
I__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_20071

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
î	
É
,__inference_sequential_1_layer_call_fn_19848
hidden_layer_1_input
unknown:I 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3:  
	unknown_4: 
	unknown_5: 
	unknown_6:
identity¢StatefulPartitionedCall¸
StatefulPartitionedCallStatefulPartitionedCallhidden_layer_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_19808o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿI: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
.
_user_specified_nameHidden-Layer-1_input
È

,__inference_Output-Layer_layer_call_fn_20100

inputs
unknown: 
	unknown_0:
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_Output-Layer_layer_call_and_return_conditional_losses_19695o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*É
serving_defaultµ
U
Hidden-Layer-1_input=
&serving_default_Hidden-Layer-1_input:0ÿÿÿÿÿÿÿÿÿI@
Output-Layer0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:

layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
»
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses

,kernel
-bias"
_tf_keras_layer
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
X
0
1
2
3
$4
%5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
æ
3trace_0
4trace_1
5trace_2
6trace_32û
,__inference_sequential_1_layer_call_fn_19721
,__inference_sequential_1_layer_call_fn_19946
,__inference_sequential_1_layer_call_fn_19967
,__inference_sequential_1_layer_call_fn_19848À
·²³
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
kwonlydefaultsª 
annotationsª *
 z3trace_0z4trace_1z5trace_2z6trace_3
Ò
7trace_0
8trace_1
9trace_2
:trace_32ç
G__inference_sequential_1_layer_call_and_return_conditional_losses_19999
G__inference_sequential_1_layer_call_and_return_conditional_losses_20031
G__inference_sequential_1_layer_call_and_return_conditional_losses_19872
G__inference_sequential_1_layer_call_and_return_conditional_losses_19896À
·²³
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
kwonlydefaultsª 
annotationsª *
 z7trace_0z8trace_1z9trace_2z:trace_3
ØBÕ
 __inference__wrapped_model_19626Hidden-Layer-1_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ª
;iter
	<decay
=learning_rate
>momentum
?rho	rmsh	rmsi	rmsj	rmsk	$rmsl	%rmsm	,rmsn	-rmso"
	optimizer
,
@serving_default"
signature_map
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
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò
Ftrace_02Õ
.__inference_Hidden-Layer-1_layer_call_fn_20040¢
²
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
annotationsª *
 zFtrace_0

Gtrace_02ð
I__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_20051¢
²
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
annotationsª *
 zGtrace_0
':%I 2Hidden-Layer-1/kernel
!: 2Hidden-Layer-1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ò
Mtrace_02Õ
.__inference_Hidden-Layer-2_layer_call_fn_20060¢
²
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
annotationsª *
 zMtrace_0

Ntrace_02ð
I__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_20071¢
²
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
annotationsª *
 zNtrace_0
':%  2Hidden-Layer-2/kernel
!: 2Hidden-Layer-2/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
ò
Ttrace_02Õ
.__inference_Hidden-Layer-3_layer_call_fn_20080¢
²
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
annotationsª *
 zTtrace_0

Utrace_02ð
I__inference_Hidden-Layer-3_layer_call_and_return_conditional_losses_20091¢
²
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
annotationsª *
 zUtrace_0
':%  2Hidden-Layer-3/kernel
!: 2Hidden-Layer-3/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Vnon_trainable_variables

Wlayers
Xmetrics
Ylayer_regularization_losses
Zlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
ð
[trace_02Ó
,__inference_Output-Layer_layer_call_fn_20100¢
²
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
annotationsª *
 z[trace_0

\trace_02î
G__inference_Output-Layer_layer_call_and_return_conditional_losses_20111¢
²
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
annotationsª *
 z\trace_0
%:# 2Output-Layer/kernel
:2Output-Layer/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_1_layer_call_fn_19721Hidden-Layer-1_input"À
·²³
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
kwonlydefaultsª 
annotationsª *
 
þBû
,__inference_sequential_1_layer_call_fn_19946inputs"À
·²³
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
kwonlydefaultsª 
annotationsª *
 
þBû
,__inference_sequential_1_layer_call_fn_19967inputs"À
·²³
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
kwonlydefaultsª 
annotationsª *
 
B
,__inference_sequential_1_layer_call_fn_19848Hidden-Layer-1_input"À
·²³
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
kwonlydefaultsª 
annotationsª *
 
B
G__inference_sequential_1_layer_call_and_return_conditional_losses_19999inputs"À
·²³
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
kwonlydefaultsª 
annotationsª *
 
B
G__inference_sequential_1_layer_call_and_return_conditional_losses_20031inputs"À
·²³
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
kwonlydefaultsª 
annotationsª *
 
§B¤
G__inference_sequential_1_layer_call_and_return_conditional_losses_19872Hidden-Layer-1_input"À
·²³
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
kwonlydefaultsª 
annotationsª *
 
§B¤
G__inference_sequential_1_layer_call_and_return_conditional_losses_19896Hidden-Layer-1_input"À
·²³
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
kwonlydefaultsª 
annotationsª *
 
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
×BÔ
#__inference_signature_wrapper_19925Hidden-Layer-1_input"
²
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
annotationsª *
 
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
âBß
.__inference_Hidden-Layer-1_layer_call_fn_20040inputs"¢
²
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
annotationsª *
 
ýBú
I__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_20051inputs"¢
²
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
annotationsª *
 
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
âBß
.__inference_Hidden-Layer-2_layer_call_fn_20060inputs"¢
²
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
annotationsª *
 
ýBú
I__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_20071inputs"¢
²
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
annotationsª *
 
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
âBß
.__inference_Hidden-Layer-3_layer_call_fn_20080inputs"¢
²
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
annotationsª *
 
ýBú
I__inference_Hidden-Layer-3_layer_call_and_return_conditional_losses_20091inputs"¢
²
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
annotationsª *
 
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
àBÝ
,__inference_Output-Layer_layer_call_fn_20100inputs"¢
²
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
annotationsª *
 
ûBø
G__inference_Output-Layer_layer_call_and_return_conditional_losses_20111inputs"¢
²
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
annotationsª *
 
N
_	variables
`	keras_api
	atotal
	bcount"
_tf_keras_metric
^
c	variables
d	keras_api
	etotal
	fcount
g
_fn_kwargs"
_tf_keras_metric
.
a0
b1"
trackable_list_wrapper
-
_	variables"
_generic_user_object
:  (2total
:  (2count
.
e0
f1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
1:/I 2!RMSprop/Hidden-Layer-1/kernel/rms
+:) 2RMSprop/Hidden-Layer-1/bias/rms
1:/  2!RMSprop/Hidden-Layer-2/kernel/rms
+:) 2RMSprop/Hidden-Layer-2/bias/rms
1:/  2!RMSprop/Hidden-Layer-3/kernel/rms
+:) 2RMSprop/Hidden-Layer-3/bias/rms
/:- 2RMSprop/Output-Layer/kernel/rms
):'2RMSprop/Output-Layer/bias/rms©
I__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_20051\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿI
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_Hidden-Layer-1_layer_call_fn_20040O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿI
ª "ÿÿÿÿÿÿÿÿÿ ©
I__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_20071\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_Hidden-Layer-2_layer_call_fn_20060O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ ©
I__inference_Hidden-Layer-3_layer_call_and_return_conditional_losses_20091\$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ 
 
.__inference_Hidden-Layer-3_layer_call_fn_20080O$%/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ §
G__inference_Output-Layer_layer_call_and_return_conditional_losses_20111\,-/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_Output-Layer_layer_call_fn_20100O,-/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ 
ª "ÿÿÿÿÿÿÿÿÿ«
 __inference__wrapped_model_19626$%,-=¢:
3¢0
.+
Hidden-Layer-1_inputÿÿÿÿÿÿÿÿÿI
ª ";ª8
6
Output-Layer&#
Output-LayerÿÿÿÿÿÿÿÿÿÃ
G__inference_sequential_1_layer_call_and_return_conditional_losses_19872x$%,-E¢B
;¢8
.+
Hidden-Layer-1_inputÿÿÿÿÿÿÿÿÿI
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
G__inference_sequential_1_layer_call_and_return_conditional_losses_19896x$%,-E¢B
;¢8
.+
Hidden-Layer-1_inputÿÿÿÿÿÿÿÿÿI
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
G__inference_sequential_1_layer_call_and_return_conditional_losses_19999j$%,-7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿI
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 µ
G__inference_sequential_1_layer_call_and_return_conditional_losses_20031j$%,-7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿI
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_sequential_1_layer_call_fn_19721k$%,-E¢B
;¢8
.+
Hidden-Layer-1_inputÿÿÿÿÿÿÿÿÿI
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_1_layer_call_fn_19848k$%,-E¢B
;¢8
.+
Hidden-Layer-1_inputÿÿÿÿÿÿÿÿÿI
p

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_1_layer_call_fn_19946]$%,-7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿI
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_sequential_1_layer_call_fn_19967]$%,-7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿI
p

 
ª "ÿÿÿÿÿÿÿÿÿÆ
#__inference_signature_wrapper_19925$%,-U¢R
¢ 
KªH
F
Hidden-Layer-1_input.+
Hidden-Layer-1_inputÿÿÿÿÿÿÿÿÿI";ª8
6
Output-Layer&#
Output-Layerÿÿÿÿÿÿÿÿÿ