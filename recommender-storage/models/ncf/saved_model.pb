??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
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
delete_old_dirsbool(?
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
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
?
/recommender_neural_network/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*@
shared_name1/recommender_neural_network/embedding/embeddings
?
Crecommender_neural_network/embedding/embeddings/Read/ReadVariableOpReadVariableOp/recommender_neural_network/embedding/embeddings*
_output_shapes
:	?2*
dtype0
?
1recommender_neural_network/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*B
shared_name31recommender_neural_network/embedding_1/embeddings
?
Erecommender_neural_network/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp1recommender_neural_network/embedding_1/embeddings*
_output_shapes
:	?*
dtype0
?
1recommender_neural_network/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?K2*B
shared_name31recommender_neural_network/embedding_2/embeddings
?
Erecommender_neural_network/embedding_2/embeddings/Read/ReadVariableOpReadVariableOp1recommender_neural_network/embedding_2/embeddings*
_output_shapes
:	?K2*
dtype0
?
1recommender_neural_network/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?K*B
shared_name31recommender_neural_network/embedding_3/embeddings
?
Erecommender_neural_network/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp1recommender_neural_network/embedding_3/embeddings*
_output_shapes
:	?K*
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
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
?
6Adam/recommender_neural_network/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*G
shared_name86Adam/recommender_neural_network/embedding/embeddings/m
?
JAdam/recommender_neural_network/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp6Adam/recommender_neural_network/embedding/embeddings/m*
_output_shapes
:	?2*
dtype0
?
8Adam/recommender_neural_network/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*I
shared_name:8Adam/recommender_neural_network/embedding_1/embeddings/m
?
LAdam/recommender_neural_network/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOp8Adam/recommender_neural_network/embedding_1/embeddings/m*
_output_shapes
:	?*
dtype0
?
8Adam/recommender_neural_network/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?K2*I
shared_name:8Adam/recommender_neural_network/embedding_2/embeddings/m
?
LAdam/recommender_neural_network/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOp8Adam/recommender_neural_network/embedding_2/embeddings/m*
_output_shapes
:	?K2*
dtype0
?
8Adam/recommender_neural_network/embedding_3/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?K*I
shared_name:8Adam/recommender_neural_network/embedding_3/embeddings/m
?
LAdam/recommender_neural_network/embedding_3/embeddings/m/Read/ReadVariableOpReadVariableOp8Adam/recommender_neural_network/embedding_3/embeddings/m*
_output_shapes
:	?K*
dtype0
?
6Adam/recommender_neural_network/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?2*G
shared_name86Adam/recommender_neural_network/embedding/embeddings/v
?
JAdam/recommender_neural_network/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp6Adam/recommender_neural_network/embedding/embeddings/v*
_output_shapes
:	?2*
dtype0
?
8Adam/recommender_neural_network/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*I
shared_name:8Adam/recommender_neural_network/embedding_1/embeddings/v
?
LAdam/recommender_neural_network/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOp8Adam/recommender_neural_network/embedding_1/embeddings/v*
_output_shapes
:	?*
dtype0
?
8Adam/recommender_neural_network/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?K2*I
shared_name:8Adam/recommender_neural_network/embedding_2/embeddings/v
?
LAdam/recommender_neural_network/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOp8Adam/recommender_neural_network/embedding_2/embeddings/v*
_output_shapes
:	?K2*
dtype0
?
8Adam/recommender_neural_network/embedding_3/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?K*I
shared_name:8Adam/recommender_neural_network/embedding_3/embeddings/v
?
LAdam/recommender_neural_network/embedding_3/embeddings/v/Read/ReadVariableOpReadVariableOp8Adam/recommender_neural_network/embedding_3/embeddings/v*
_output_shapes
:	?K*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?!
value?!B?! B?!
?
user_embedding
	user_bias
movie_embedding

movie_bias
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
?
iter

 beta_1

!beta_2
	"decay
#learning_ratemBmCmDmEvFvGvHvI

0
1
2
3

0
1
2
3
 
?
	variables
trainable_variables
$non_trainable_variables
regularization_losses

%layers
&metrics
'layer_regularization_losses
(layer_metrics
 
yw
VARIABLE_VALUE/recommender_neural_network/embedding/embeddings4user_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
	variables
trainable_variables
)non_trainable_variables
regularization_losses

*layers
+metrics
,layer_regularization_losses
-layer_metrics
vt
VARIABLE_VALUE1recommender_neural_network/embedding_1/embeddings/user_bias/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
	variables
trainable_variables
.non_trainable_variables
regularization_losses

/layers
0metrics
1layer_regularization_losses
2layer_metrics
|z
VARIABLE_VALUE1recommender_neural_network/embedding_2/embeddings5movie_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
	variables
trainable_variables
3non_trainable_variables
regularization_losses

4layers
5metrics
6layer_regularization_losses
7layer_metrics
wu
VARIABLE_VALUE1recommender_neural_network/embedding_3/embeddings0movie_bias/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
?
	variables
trainable_variables
8non_trainable_variables
regularization_losses

9layers
:metrics
;layer_regularization_losses
<layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

0
1
2
3

=0
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
	>total
	?count
@	variables
A	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

>0
?1

@	variables
??
VARIABLE_VALUE6Adam/recommender_neural_network/embedding/embeddings/mPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/recommender_neural_network/embedding_1/embeddings/mKuser_bias/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/recommender_neural_network/embedding_2/embeddings/mQmovie_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/recommender_neural_network/embedding_3/embeddings/mLmovie_bias/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE6Adam/recommender_neural_network/embedding/embeddings/vPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/recommender_neural_network/embedding_1/embeddings/vKuser_bias/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/recommender_neural_network/embedding_2/embeddings/vQmovie_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE8Adam/recommender_neural_network/embedding_3/embeddings/vLmovie_bias/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0	*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1/recommender_neural_network/embedding/embeddings1recommender_neural_network/embedding_1/embeddings1recommender_neural_network/embedding_2/embeddings1recommender_neural_network/embedding_3/embeddings*
Tin	
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_9456
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameCrecommender_neural_network/embedding/embeddings/Read/ReadVariableOpErecommender_neural_network/embedding_1/embeddings/Read/ReadVariableOpErecommender_neural_network/embedding_2/embeddings/Read/ReadVariableOpErecommender_neural_network/embedding_3/embeddings/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpJAdam/recommender_neural_network/embedding/embeddings/m/Read/ReadVariableOpLAdam/recommender_neural_network/embedding_1/embeddings/m/Read/ReadVariableOpLAdam/recommender_neural_network/embedding_2/embeddings/m/Read/ReadVariableOpLAdam/recommender_neural_network/embedding_3/embeddings/m/Read/ReadVariableOpJAdam/recommender_neural_network/embedding/embeddings/v/Read/ReadVariableOpLAdam/recommender_neural_network/embedding_1/embeddings/v/Read/ReadVariableOpLAdam/recommender_neural_network/embedding_2/embeddings/v/Read/ReadVariableOpLAdam/recommender_neural_network/embedding_3/embeddings/v/Read/ReadVariableOpConst* 
Tin
2	*
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
GPU 2J 8? *&
f!R
__inference__traced_save_9646
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename/recommender_neural_network/embedding/embeddings1recommender_neural_network/embedding_1/embeddings1recommender_neural_network/embedding_2/embeddings1recommender_neural_network/embedding_3/embeddings	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount6Adam/recommender_neural_network/embedding/embeddings/m8Adam/recommender_neural_network/embedding_1/embeddings/m8Adam/recommender_neural_network/embedding_2/embeddings/m8Adam/recommender_neural_network/embedding_3/embeddings/m6Adam/recommender_neural_network/embedding/embeddings/v8Adam/recommender_neural_network/embedding_1/embeddings/v8Adam/recommender_neural_network/embedding_2/embeddings/v8Adam/recommender_neural_network/embedding_3/embeddings/v*
Tin
2*
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
GPU 2J 8? *)
f$R"
 __inference__traced_restore_9713??
?
?
E__inference_embedding_2_layer_call_and_return_conditional_losses_9318

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?K2*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????22
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:?????????22
embedding_lookup/Identity?
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?K2*
dtype02U
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
Drecommender_neural_network/embedding_2/embeddings/Regularizer/SquareSquare[recommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?K22F
Drecommender_neural_network/embedding_2/embeddings/Regularizer/Square?
Crecommender_neural_network/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2E
Crecommender_neural_network/embedding_2/embeddings/Regularizer/Const?
Arecommender_neural_network/embedding_2/embeddings/Regularizer/SumSumHrecommender_neural_network/embedding_2/embeddings/Regularizer/Square:y:0Lrecommender_neural_network/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2C
Arecommender_neural_network/embedding_2/embeddings/Regularizer/Sum?
Crecommender_neural_network/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52E
Crecommender_neural_network/embedding_2/embeddings/Regularizer/mul/x?
Arecommender_neural_network/embedding_2/embeddings/Regularizer/mulMulLrecommender_neural_network/embedding_2/embeddings/Regularizer/mul/x:output:0Jrecommender_neural_network/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2C
Arecommender_neural_network/embedding_2/embeddings/Regularizer/mul?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpT^recommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2?
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOpSrecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_9239
input_1	Q
Mrecommender_neural_network_embedding_embedding_lookup_readvariableop_resourceS
Orecommender_neural_network_embedding_1_embedding_lookup_readvariableop_resourceS
Orecommender_neural_network_embedding_2_embedding_lookup_readvariableop_resourceS
Orecommender_neural_network_embedding_3_embedding_lookup_readvariableop_resource
identity??Drecommender_neural_network/embedding/embedding_lookup/ReadVariableOp?Frecommender_neural_network/embedding_1/embedding_lookup/ReadVariableOp?Frecommender_neural_network/embedding_2/embedding_lookup/ReadVariableOp?Frecommender_neural_network/embedding_3/embedding_lookup/ReadVariableOp?
.recommender_neural_network/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.recommender_neural_network/strided_slice/stack?
0recommender_neural_network/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0recommender_neural_network/strided_slice/stack_1?
0recommender_neural_network/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0recommender_neural_network/strided_slice/stack_2?
(recommender_neural_network/strided_sliceStridedSliceinput_17recommender_neural_network/strided_slice/stack:output:09recommender_neural_network/strided_slice/stack_1:output:09recommender_neural_network/strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2*
(recommender_neural_network/strided_slice?
Drecommender_neural_network/embedding/embedding_lookup/ReadVariableOpReadVariableOpMrecommender_neural_network_embedding_embedding_lookup_readvariableop_resource*
_output_shapes
:	?2*
dtype02F
Drecommender_neural_network/embedding/embedding_lookup/ReadVariableOp?
:recommender_neural_network/embedding/embedding_lookup/axisConst*W
_classM
KIloc:@recommender_neural_network/embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2<
:recommender_neural_network/embedding/embedding_lookup/axis?
5recommender_neural_network/embedding/embedding_lookupGatherV2Lrecommender_neural_network/embedding/embedding_lookup/ReadVariableOp:value:01recommender_neural_network/strided_slice:output:0Crecommender_neural_network/embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*W
_classM
KIloc:@recommender_neural_network/embedding/embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????227
5recommender_neural_network/embedding/embedding_lookup?
>recommender_neural_network/embedding/embedding_lookup/IdentityIdentity>recommender_neural_network/embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:?????????22@
>recommender_neural_network/embedding/embedding_lookup/Identity?
0recommender_neural_network/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        22
0recommender_neural_network/strided_slice_1/stack?
2recommender_neural_network/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2recommender_neural_network/strided_slice_1/stack_1?
2recommender_neural_network/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2recommender_neural_network/strided_slice_1/stack_2?
*recommender_neural_network/strided_slice_1StridedSliceinput_19recommender_neural_network/strided_slice_1/stack:output:0;recommender_neural_network/strided_slice_1/stack_1:output:0;recommender_neural_network/strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2,
*recommender_neural_network/strided_slice_1?
Frecommender_neural_network/embedding_1/embedding_lookup/ReadVariableOpReadVariableOpOrecommender_neural_network_embedding_1_embedding_lookup_readvariableop_resource*
_output_shapes
:	?*
dtype02H
Frecommender_neural_network/embedding_1/embedding_lookup/ReadVariableOp?
<recommender_neural_network/embedding_1/embedding_lookup/axisConst*Y
_classO
MKloc:@recommender_neural_network/embedding_1/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2>
<recommender_neural_network/embedding_1/embedding_lookup/axis?
7recommender_neural_network/embedding_1/embedding_lookupGatherV2Nrecommender_neural_network/embedding_1/embedding_lookup/ReadVariableOp:value:03recommender_neural_network/strided_slice_1:output:0Erecommender_neural_network/embedding_1/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*Y
_classO
MKloc:@recommender_neural_network/embedding_1/embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????29
7recommender_neural_network/embedding_1/embedding_lookup?
@recommender_neural_network/embedding_1/embedding_lookup/IdentityIdentity@recommender_neural_network/embedding_1/embedding_lookup:output:0*
T0*'
_output_shapes
:?????????2B
@recommender_neural_network/embedding_1/embedding_lookup/Identity?
0recommender_neural_network/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0recommender_neural_network/strided_slice_2/stack?
2recommender_neural_network/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2recommender_neural_network/strided_slice_2/stack_1?
2recommender_neural_network/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2recommender_neural_network/strided_slice_2/stack_2?
*recommender_neural_network/strided_slice_2StridedSliceinput_19recommender_neural_network/strided_slice_2/stack:output:0;recommender_neural_network/strided_slice_2/stack_1:output:0;recommender_neural_network/strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2,
*recommender_neural_network/strided_slice_2?
Frecommender_neural_network/embedding_2/embedding_lookup/ReadVariableOpReadVariableOpOrecommender_neural_network_embedding_2_embedding_lookup_readvariableop_resource*
_output_shapes
:	?K2*
dtype02H
Frecommender_neural_network/embedding_2/embedding_lookup/ReadVariableOp?
<recommender_neural_network/embedding_2/embedding_lookup/axisConst*Y
_classO
MKloc:@recommender_neural_network/embedding_2/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2>
<recommender_neural_network/embedding_2/embedding_lookup/axis?
7recommender_neural_network/embedding_2/embedding_lookupGatherV2Nrecommender_neural_network/embedding_2/embedding_lookup/ReadVariableOp:value:03recommender_neural_network/strided_slice_2:output:0Erecommender_neural_network/embedding_2/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*Y
_classO
MKloc:@recommender_neural_network/embedding_2/embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????229
7recommender_neural_network/embedding_2/embedding_lookup?
@recommender_neural_network/embedding_2/embedding_lookup/IdentityIdentity@recommender_neural_network/embedding_2/embedding_lookup:output:0*
T0*'
_output_shapes
:?????????22B
@recommender_neural_network/embedding_2/embedding_lookup/Identity?
0recommender_neural_network/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       22
0recommender_neural_network/strided_slice_3/stack?
2recommender_neural_network/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       24
2recommender_neural_network/strided_slice_3/stack_1?
2recommender_neural_network/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      24
2recommender_neural_network/strided_slice_3/stack_2?
*recommender_neural_network/strided_slice_3StridedSliceinput_19recommender_neural_network/strided_slice_3/stack:output:0;recommender_neural_network/strided_slice_3/stack_1:output:0;recommender_neural_network/strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2,
*recommender_neural_network/strided_slice_3?
Frecommender_neural_network/embedding_3/embedding_lookup/ReadVariableOpReadVariableOpOrecommender_neural_network_embedding_3_embedding_lookup_readvariableop_resource*
_output_shapes
:	?K*
dtype02H
Frecommender_neural_network/embedding_3/embedding_lookup/ReadVariableOp?
<recommender_neural_network/embedding_3/embedding_lookup/axisConst*Y
_classO
MKloc:@recommender_neural_network/embedding_3/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2>
<recommender_neural_network/embedding_3/embedding_lookup/axis?
7recommender_neural_network/embedding_3/embedding_lookupGatherV2Nrecommender_neural_network/embedding_3/embedding_lookup/ReadVariableOp:value:03recommender_neural_network/strided_slice_3:output:0Erecommender_neural_network/embedding_3/embedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*Y
_classO
MKloc:@recommender_neural_network/embedding_3/embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????29
7recommender_neural_network/embedding_3/embedding_lookup?
@recommender_neural_network/embedding_3/embedding_lookup/IdentityIdentity@recommender_neural_network/embedding_3/embedding_lookup:output:0*
T0*'
_output_shapes
:?????????2B
@recommender_neural_network/embedding_3/embedding_lookup/Identity?
)recommender_neural_network/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB"       2+
)recommender_neural_network/Tensordot/axes?
)recommender_neural_network/Tensordot/freeConst*
_output_shapes
: *
dtype0*
valueB 2+
)recommender_neural_network/Tensordot/free?
*recommender_neural_network/Tensordot/ShapeShapeGrecommender_neural_network/embedding/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:2,
*recommender_neural_network/Tensordot/Shape?
2recommender_neural_network/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2recommender_neural_network/Tensordot/GatherV2/axis?
-recommender_neural_network/Tensordot/GatherV2GatherV23recommender_neural_network/Tensordot/Shape:output:02recommender_neural_network/Tensordot/free:output:0;recommender_neural_network/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2/
-recommender_neural_network/Tensordot/GatherV2?
4recommender_neural_network/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4recommender_neural_network/Tensordot/GatherV2_1/axis?
/recommender_neural_network/Tensordot/GatherV2_1GatherV23recommender_neural_network/Tensordot/Shape:output:02recommender_neural_network/Tensordot/axes:output:0=recommender_neural_network/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:21
/recommender_neural_network/Tensordot/GatherV2_1?
*recommender_neural_network/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2,
*recommender_neural_network/Tensordot/Const?
)recommender_neural_network/Tensordot/ProdProd6recommender_neural_network/Tensordot/GatherV2:output:03recommender_neural_network/Tensordot/Const:output:0*
T0*
_output_shapes
: 2+
)recommender_neural_network/Tensordot/Prod?
,recommender_neural_network/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,recommender_neural_network/Tensordot/Const_1?
+recommender_neural_network/Tensordot/Prod_1Prod8recommender_neural_network/Tensordot/GatherV2_1:output:05recommender_neural_network/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2-
+recommender_neural_network/Tensordot/Prod_1?
0recommender_neural_network/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0recommender_neural_network/Tensordot/concat/axis?
+recommender_neural_network/Tensordot/concatConcatV22recommender_neural_network/Tensordot/free:output:02recommender_neural_network/Tensordot/axes:output:09recommender_neural_network/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2-
+recommender_neural_network/Tensordot/concat?
*recommender_neural_network/Tensordot/stackPack2recommender_neural_network/Tensordot/Prod:output:04recommender_neural_network/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2,
*recommender_neural_network/Tensordot/stack?
.recommender_neural_network/Tensordot/transpose	TransposeGrecommender_neural_network/embedding/embedding_lookup/Identity:output:04recommender_neural_network/Tensordot/concat:output:0*
T0*'
_output_shapes
:?????????220
.recommender_neural_network/Tensordot/transpose?
,recommender_neural_network/Tensordot/ReshapeReshape2recommender_neural_network/Tensordot/transpose:y:03recommender_neural_network/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2.
,recommender_neural_network/Tensordot/Reshape?
+recommender_neural_network/Tensordot/axes_1Const*
_output_shapes
:*
dtype0*
valueB"       2-
+recommender_neural_network/Tensordot/axes_1?
+recommender_neural_network/Tensordot/free_1Const*
_output_shapes
: *
dtype0*
valueB 2-
+recommender_neural_network/Tensordot/free_1?
,recommender_neural_network/Tensordot/Shape_1ShapeIrecommender_neural_network/embedding_2/embedding_lookup/Identity:output:0*
T0*
_output_shapes
:2.
,recommender_neural_network/Tensordot/Shape_1?
4recommender_neural_network/Tensordot/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4recommender_neural_network/Tensordot/GatherV2_2/axis?
/recommender_neural_network/Tensordot/GatherV2_2GatherV25recommender_neural_network/Tensordot/Shape_1:output:04recommender_neural_network/Tensordot/free_1:output:0=recommender_neural_network/Tensordot/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 21
/recommender_neural_network/Tensordot/GatherV2_2?
4recommender_neural_network/Tensordot/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 26
4recommender_neural_network/Tensordot/GatherV2_3/axis?
/recommender_neural_network/Tensordot/GatherV2_3GatherV25recommender_neural_network/Tensordot/Shape_1:output:04recommender_neural_network/Tensordot/axes_1:output:0=recommender_neural_network/Tensordot/GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:21
/recommender_neural_network/Tensordot/GatherV2_3?
,recommender_neural_network/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2.
,recommender_neural_network/Tensordot/Const_2?
+recommender_neural_network/Tensordot/Prod_2Prod8recommender_neural_network/Tensordot/GatherV2_2:output:05recommender_neural_network/Tensordot/Const_2:output:0*
T0*
_output_shapes
: 2-
+recommender_neural_network/Tensordot/Prod_2?
,recommender_neural_network/Tensordot/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2.
,recommender_neural_network/Tensordot/Const_3?
+recommender_neural_network/Tensordot/Prod_3Prod8recommender_neural_network/Tensordot/GatherV2_3:output:05recommender_neural_network/Tensordot/Const_3:output:0*
T0*
_output_shapes
: 2-
+recommender_neural_network/Tensordot/Prod_3?
2recommender_neural_network/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2recommender_neural_network/Tensordot/concat_1/axis?
-recommender_neural_network/Tensordot/concat_1ConcatV24recommender_neural_network/Tensordot/axes_1:output:04recommender_neural_network/Tensordot/free_1:output:0;recommender_neural_network/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2/
-recommender_neural_network/Tensordot/concat_1?
,recommender_neural_network/Tensordot/stack_1Pack4recommender_neural_network/Tensordot/Prod_3:output:04recommender_neural_network/Tensordot/Prod_2:output:0*
N*
T0*
_output_shapes
:2.
,recommender_neural_network/Tensordot/stack_1?
0recommender_neural_network/Tensordot/transpose_1	TransposeIrecommender_neural_network/embedding_2/embedding_lookup/Identity:output:06recommender_neural_network/Tensordot/concat_1:output:0*
T0*'
_output_shapes
:?????????222
0recommender_neural_network/Tensordot/transpose_1?
.recommender_neural_network/Tensordot/Reshape_1Reshape4recommender_neural_network/Tensordot/transpose_1:y:05recommender_neural_network/Tensordot/stack_1:output:0*
T0*0
_output_shapes
:??????????????????20
.recommender_neural_network/Tensordot/Reshape_1?
+recommender_neural_network/Tensordot/MatMulMatMul5recommender_neural_network/Tensordot/Reshape:output:07recommender_neural_network/Tensordot/Reshape_1:output:0*
T0*0
_output_shapes
:??????????????????2-
+recommender_neural_network/Tensordot/MatMul?
2recommender_neural_network/Tensordot/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 24
2recommender_neural_network/Tensordot/concat_2/axis?
-recommender_neural_network/Tensordot/concat_2ConcatV26recommender_neural_network/Tensordot/GatherV2:output:08recommender_neural_network/Tensordot/GatherV2_2:output:0;recommender_neural_network/Tensordot/concat_2/axis:output:0*
N*
T0*
_output_shapes
: 2/
-recommender_neural_network/Tensordot/concat_2?
$recommender_neural_network/TensordotReshape5recommender_neural_network/Tensordot/MatMul:product:06recommender_neural_network/Tensordot/concat_2:output:0*
T0*
_output_shapes
: 2&
$recommender_neural_network/Tensordot?
recommender_neural_network/addAddV2-recommender_neural_network/Tensordot:output:0Irecommender_neural_network/embedding_1/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2 
recommender_neural_network/add?
 recommender_neural_network/add_1AddV2"recommender_neural_network/add:z:0Irecommender_neural_network/embedding_3/embedding_lookup/Identity:output:0*
T0*'
_output_shapes
:?????????2"
 recommender_neural_network/add_1?
"recommender_neural_network/SigmoidSigmoid$recommender_neural_network/add_1:z:0*
T0*'
_output_shapes
:?????????2$
"recommender_neural_network/Sigmoid?
IdentityIdentity&recommender_neural_network/Sigmoid:y:0E^recommender_neural_network/embedding/embedding_lookup/ReadVariableOpG^recommender_neural_network/embedding_1/embedding_lookup/ReadVariableOpG^recommender_neural_network/embedding_2/embedding_lookup/ReadVariableOpG^recommender_neural_network/embedding_3/embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2?
Drecommender_neural_network/embedding/embedding_lookup/ReadVariableOpDrecommender_neural_network/embedding/embedding_lookup/ReadVariableOp2?
Frecommender_neural_network/embedding_1/embedding_lookup/ReadVariableOpFrecommender_neural_network/embedding_1/embedding_lookup/ReadVariableOp2?
Frecommender_neural_network/embedding_2/embedding_lookup/ReadVariableOpFrecommender_neural_network/embedding_2/embedding_lookup/ReadVariableOp2?
Frecommender_neural_network/embedding_3/embedding_lookup/ReadVariableOpFrecommender_neural_network/embedding_3/embedding_lookup/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
p
*__inference_embedding_1_layer_call_fn_9500

inputs	
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_92872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_embedding_1_layer_call_and_return_conditional_losses_9287

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
p
*__inference_embedding_3_layer_call_fn_9544

inputs	
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_93432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_embedding_layer_call_and_return_conditional_losses_9477

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?2*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????22
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:?????????22
embedding_lookup/Identity?
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?2*
dtype02S
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp?
Brecommender_neural_network/embedding/embeddings/Regularizer/SquareSquareYrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?22D
Brecommender_neural_network/embedding/embeddings/Regularizer/Square?
Arecommender_neural_network/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Arecommender_neural_network/embedding/embeddings/Regularizer/Const?
?recommender_neural_network/embedding/embeddings/Regularizer/SumSumFrecommender_neural_network/embedding/embeddings/Regularizer/Square:y:0Jrecommender_neural_network/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?recommender_neural_network/embedding/embeddings/Regularizer/Sum?
Arecommender_neural_network/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52C
Arecommender_neural_network/embedding/embeddings/Regularizer/mul/x?
?recommender_neural_network/embedding/embeddings/Regularizer/mulMulJrecommender_neural_network/embedding/embeddings/Regularizer/mul/x:output:0Hrecommender_neural_network/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?recommender_neural_network/embedding/embeddings/Regularizer/mul?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpR^recommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2?
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOpQrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
9__inference_recommender_neural_network_layer_call_fn_9421
input_1	
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *]
fXRV
T__inference_recommender_neural_network_layer_call_and_return_conditional_losses_94072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
E__inference_embedding_2_layer_call_and_return_conditional_losses_9521

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?K2*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????22
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:?????????22
embedding_lookup/Identity?
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?K2*
dtype02U
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
Drecommender_neural_network/embedding_2/embeddings/Regularizer/SquareSquare[recommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?K22F
Drecommender_neural_network/embedding_2/embeddings/Regularizer/Square?
Crecommender_neural_network/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2E
Crecommender_neural_network/embedding_2/embeddings/Regularizer/Const?
Arecommender_neural_network/embedding_2/embeddings/Regularizer/SumSumHrecommender_neural_network/embedding_2/embeddings/Regularizer/Square:y:0Lrecommender_neural_network/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2C
Arecommender_neural_network/embedding_2/embeddings/Regularizer/Sum?
Crecommender_neural_network/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52E
Crecommender_neural_network/embedding_2/embeddings/Regularizer/mul/x?
Arecommender_neural_network/embedding_2/embeddings/Regularizer/mulMulLrecommender_neural_network/embedding_2/embeddings/Regularizer/mul/x:output:0Jrecommender_neural_network/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2C
Arecommender_neural_network/embedding_2/embeddings/Regularizer/mul?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpT^recommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2?
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOpSrecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
E__inference_embedding_3_layer_call_and_return_conditional_losses_9537

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?K*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
"__inference_signature_wrapper_9456
input_1	
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_92392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
p
*__inference_embedding_2_layer_call_fn_9528

inputs	
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_93182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?X
?
 __inference__traced_restore_9713
file_prefixD
@assignvariableop_recommender_neural_network_embedding_embeddingsH
Dassignvariableop_1_recommender_neural_network_embedding_1_embeddingsH
Dassignvariableop_2_recommender_neural_network_embedding_2_embeddingsH
Dassignvariableop_3_recommender_neural_network_embedding_3_embeddings 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_countN
Jassignvariableop_11_adam_recommender_neural_network_embedding_embeddings_mP
Lassignvariableop_12_adam_recommender_neural_network_embedding_1_embeddings_mP
Lassignvariableop_13_adam_recommender_neural_network_embedding_2_embeddings_mP
Lassignvariableop_14_adam_recommender_neural_network_embedding_3_embeddings_mN
Jassignvariableop_15_adam_recommender_neural_network_embedding_embeddings_vP
Lassignvariableop_16_adam_recommender_neural_network_embedding_1_embeddings_vP
Lassignvariableop_17_adam_recommender_neural_network_embedding_2_embeddings_vP
Lassignvariableop_18_adam_recommender_neural_network_embedding_3_embeddings_v
identity_20??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B4user_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB/user_bias/embeddings/.ATTRIBUTES/VARIABLE_VALUEB5movie_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB0movie_bias/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKuser_bias/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQmovie_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLmovie_bias/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKuser_bias/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQmovie_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLmovie_bias/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*d
_output_shapesR
P::::::::::::::::::::*"
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp@assignvariableop_recommender_neural_network_embedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpDassignvariableop_1_recommender_neural_network_embedding_1_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpDassignvariableop_2_recommender_neural_network_embedding_2_embeddingsIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpDassignvariableop_3_recommender_neural_network_embedding_3_embeddingsIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpJassignvariableop_11_adam_recommender_neural_network_embedding_embeddings_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpLassignvariableop_12_adam_recommender_neural_network_embedding_1_embeddings_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpLassignvariableop_13_adam_recommender_neural_network_embedding_2_embeddings_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpLassignvariableop_14_adam_recommender_neural_network_embedding_3_embeddings_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpJassignvariableop_15_adam_recommender_neural_network_embedding_embeddings_vIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpLassignvariableop_16_adam_recommender_neural_network_embedding_1_embeddings_vIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpLassignvariableop_17_adam_recommender_neural_network_embedding_2_embeddings_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpLassignvariableop_18_adam_recommender_neural_network_embedding_3_embeddings_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_189
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19?
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2$
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
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
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
?
?
C__inference_embedding_layer_call_and_return_conditional_losses_9262

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?2*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????22
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:?????????22
embedding_lookup/Identity?
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?2*
dtype02S
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp?
Brecommender_neural_network/embedding/embeddings/Regularizer/SquareSquareYrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?22D
Brecommender_neural_network/embedding/embeddings/Regularizer/Square?
Arecommender_neural_network/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Arecommender_neural_network/embedding/embeddings/Regularizer/Const?
?recommender_neural_network/embedding/embeddings/Regularizer/SumSumFrecommender_neural_network/embedding/embeddings/Regularizer/Square:y:0Jrecommender_neural_network/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?recommender_neural_network/embedding/embeddings/Regularizer/Sum?
Arecommender_neural_network/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52C
Arecommender_neural_network/embedding/embeddings/Regularizer/mul/x?
?recommender_neural_network/embedding/embeddings/Regularizer/mulMulJrecommender_neural_network/embedding/embeddings/Regularizer/mul/x:output:0Hrecommender_neural_network/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?recommender_neural_network/embedding/embeddings/Regularizer/mul?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpR^recommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2?
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOpQrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_0_9555^
Zrecommender_neural_network_embedding_embeddings_regularizer_square_readvariableop_resource
identity??Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp?
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpZrecommender_neural_network_embedding_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	?2*
dtype02S
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp?
Brecommender_neural_network/embedding/embeddings/Regularizer/SquareSquareYrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?22D
Brecommender_neural_network/embedding/embeddings/Regularizer/Square?
Arecommender_neural_network/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Arecommender_neural_network/embedding/embeddings/Regularizer/Const?
?recommender_neural_network/embedding/embeddings/Regularizer/SumSumFrecommender_neural_network/embedding/embeddings/Regularizer/Square:y:0Jrecommender_neural_network/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?recommender_neural_network/embedding/embeddings/Regularizer/Sum?
Arecommender_neural_network/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52C
Arecommender_neural_network/embedding/embeddings/Regularizer/mul/x?
?recommender_neural_network/embedding/embeddings/Regularizer/mulMulJrecommender_neural_network/embedding/embeddings/Regularizer/mul/x:output:0Hrecommender_neural_network/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?recommender_neural_network/embedding/embeddings/Regularizer/mul?
IdentityIdentityCrecommender_neural_network/embedding/embeddings/Regularizer/mul:z:0R^recommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2?
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOpQrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp
?7
?
__inference__traced_save_9646
file_prefixN
Jsavev2_recommender_neural_network_embedding_embeddings_read_readvariableopP
Lsavev2_recommender_neural_network_embedding_1_embeddings_read_readvariableopP
Lsavev2_recommender_neural_network_embedding_2_embeddings_read_readvariableopP
Lsavev2_recommender_neural_network_embedding_3_embeddings_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopU
Qsavev2_adam_recommender_neural_network_embedding_embeddings_m_read_readvariableopW
Ssavev2_adam_recommender_neural_network_embedding_1_embeddings_m_read_readvariableopW
Ssavev2_adam_recommender_neural_network_embedding_2_embeddings_m_read_readvariableopW
Ssavev2_adam_recommender_neural_network_embedding_3_embeddings_m_read_readvariableopU
Qsavev2_adam_recommender_neural_network_embedding_embeddings_v_read_readvariableopW
Ssavev2_adam_recommender_neural_network_embedding_1_embeddings_v_read_readvariableopW
Ssavev2_adam_recommender_neural_network_embedding_2_embeddings_v_read_readvariableopW
Ssavev2_adam_recommender_neural_network_embedding_3_embeddings_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?	B?	B4user_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB/user_bias/embeddings/.ATTRIBUTES/VARIABLE_VALUEB5movie_embedding/embeddings/.ATTRIBUTES/VARIABLE_VALUEB0movie_bias/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBKuser_bias/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQmovie_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLmovie_bias/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPuser_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBKuser_bias/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQmovie_embedding/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLmovie_bias/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*;
value2B0B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Jsavev2_recommender_neural_network_embedding_embeddings_read_readvariableopLsavev2_recommender_neural_network_embedding_1_embeddings_read_readvariableopLsavev2_recommender_neural_network_embedding_2_embeddings_read_readvariableopLsavev2_recommender_neural_network_embedding_3_embeddings_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopQsavev2_adam_recommender_neural_network_embedding_embeddings_m_read_readvariableopSsavev2_adam_recommender_neural_network_embedding_1_embeddings_m_read_readvariableopSsavev2_adam_recommender_neural_network_embedding_2_embeddings_m_read_readvariableopSsavev2_adam_recommender_neural_network_embedding_3_embeddings_m_read_readvariableopQsavev2_adam_recommender_neural_network_embedding_embeddings_v_read_readvariableopSsavev2_adam_recommender_neural_network_embedding_1_embeddings_v_read_readvariableopSsavev2_adam_recommender_neural_network_embedding_2_embeddings_v_read_readvariableopSsavev2_adam_recommender_neural_network_embedding_3_embeddings_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *"
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?2:	?:	?K2:	?K: : : : : : : :	?2:	?:	?K2:	?K:	?2:	?:	?K2:	?K: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?2:%!

_output_shapes
:	?:%!

_output_shapes
:	?K2:%!

_output_shapes
:	?K:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?2:%!

_output_shapes
:	?:%!

_output_shapes
:	?K2:%!

_output_shapes
:	?K:%!

_output_shapes
:	?2:%!

_output_shapes
:	?:%!

_output_shapes
:	?K2:%!

_output_shapes
:	?K:

_output_shapes
: 
?

?
E__inference_embedding_1_layer_call_and_return_conditional_losses_9493

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_9566`
\recommender_neural_network_embedding_2_embeddings_regularizer_square_readvariableop_resource
identity??Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp\recommender_neural_network_embedding_2_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	?K2*
dtype02U
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
Drecommender_neural_network/embedding_2/embeddings/Regularizer/SquareSquare[recommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?K22F
Drecommender_neural_network/embedding_2/embeddings/Regularizer/Square?
Crecommender_neural_network/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2E
Crecommender_neural_network/embedding_2/embeddings/Regularizer/Const?
Arecommender_neural_network/embedding_2/embeddings/Regularizer/SumSumHrecommender_neural_network/embedding_2/embeddings/Regularizer/Square:y:0Lrecommender_neural_network/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2C
Arecommender_neural_network/embedding_2/embeddings/Regularizer/Sum?
Crecommender_neural_network/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52E
Crecommender_neural_network/embedding_2/embeddings/Regularizer/mul/x?
Arecommender_neural_network/embedding_2/embeddings/Regularizer/mulMulLrecommender_neural_network/embedding_2/embeddings/Regularizer/mul/x:output:0Jrecommender_neural_network/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2C
Arecommender_neural_network/embedding_2/embeddings/Regularizer/mul?
IdentityIdentityErecommender_neural_network/embedding_2/embeddings/Regularizer/mul:z:0T^recommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2?
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOpSrecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp
?t
?
T__inference_recommender_neural_network_layer_call_and_return_conditional_losses_9407
input_1	
embedding_9271
embedding_1_9296
embedding_2_9327
embedding_3_9352
identity??!embedding/StatefulPartitionedCall?#embedding_1/StatefulPartitionedCall?#embedding_2/StatefulPartitionedCall?#embedding_3/StatefulPartitionedCall?Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp?Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice?
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_9271*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_92622#
!embedding/StatefulPartitionedCall
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1?
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_1_9296*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_92872%
#embedding_1/StatefulPartitionedCall
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinput_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2?
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_2_9327*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_93182%
#embedding_2/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceinput_1strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3?
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_3_9352*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_93432%
#embedding_3/StatefulPartitionedCallq
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/axesc
Tensordot/freeConst*
_output_shapes
: *
dtype0*
valueB 2
Tensordot/free|
Tensordot/ShapeShape*embedding/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	Transpose*embedding/StatefulPartitionedCall:output:0Tensordot/concat:output:0*
T0*'
_output_shapes
:?????????22
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshapeu
Tensordot/axes_1Const*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/axes_1g
Tensordot/free_1Const*
_output_shapes
: *
dtype0*
valueB 2
Tensordot/free_1?
Tensordot/Shape_1Shape,embedding_2/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2
Tensordot/Shape_1x
Tensordot/GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_2/axis?
Tensordot/GatherV2_2GatherV2Tensordot/Shape_1:output:0Tensordot/free_1:output:0"Tensordot/GatherV2_2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
Tensordot/GatherV2_2x
Tensordot/GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_3/axis?
Tensordot/GatherV2_3GatherV2Tensordot/Shape_1:output:0Tensordot/axes_1:output:0"Tensordot/GatherV2_3/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_3p
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_2?
Tensordot/Prod_2ProdTensordot/GatherV2_2:output:0Tensordot/Const_2:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_2p
Tensordot/Const_3Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_3?
Tensordot/Prod_3ProdTensordot/GatherV2_3:output:0Tensordot/Const_3:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_3t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/axes_1:output:0Tensordot/free_1:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
Tensordot/stack_1PackTensordot/Prod_3:output:0Tensordot/Prod_2:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack_1?
Tensordot/transpose_1	Transpose,embedding_2/StatefulPartitionedCall:output:0Tensordot/concat_1:output:0*
T0*'
_output_shapes
:?????????22
Tensordot/transpose_1?
Tensordot/Reshape_1ReshapeTensordot/transpose_1:y:0Tensordot/stack_1:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape_1?
Tensordot/MatMulMatMulTensordot/Reshape:output:0Tensordot/Reshape_1:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/MatMult
Tensordot/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_2/axis?
Tensordot/concat_2ConcatV2Tensordot/GatherV2:output:0Tensordot/GatherV2_2:output:0 Tensordot/concat_2/axis:output:0*
N*
T0*
_output_shapes
: 2
Tensordot/concat_2{
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_2:output:0*
T0*
_output_shapes
: 2
	Tensordot?
addAddV2Tensordot:output:0,embedding_1/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
add?
add_1AddV2add:z:0,embedding_3/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
add_1Z
SigmoidSigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_9271*
_output_shapes
:	?2*
dtype02S
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp?
Brecommender_neural_network/embedding/embeddings/Regularizer/SquareSquareYrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?22D
Brecommender_neural_network/embedding/embeddings/Regularizer/Square?
Arecommender_neural_network/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2C
Arecommender_neural_network/embedding/embeddings/Regularizer/Const?
?recommender_neural_network/embedding/embeddings/Regularizer/SumSumFrecommender_neural_network/embedding/embeddings/Regularizer/Square:y:0Jrecommender_neural_network/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2A
?recommender_neural_network/embedding/embeddings/Regularizer/Sum?
Arecommender_neural_network/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52C
Arecommender_neural_network/embedding/embeddings/Regularizer/mul/x?
?recommender_neural_network/embedding/embeddings/Regularizer/mulMulJrecommender_neural_network/embedding/embeddings/Regularizer/mul/x:output:0Hrecommender_neural_network/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2A
?recommender_neural_network/embedding/embeddings/Regularizer/mul?
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_9327*
_output_shapes
:	?K2*
dtype02U
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp?
Drecommender_neural_network/embedding_2/embeddings/Regularizer/SquareSquare[recommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	?K22F
Drecommender_neural_network/embedding_2/embeddings/Regularizer/Square?
Crecommender_neural_network/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2E
Crecommender_neural_network/embedding_2/embeddings/Regularizer/Const?
Arecommender_neural_network/embedding_2/embeddings/Regularizer/SumSumHrecommender_neural_network/embedding_2/embeddings/Regularizer/Square:y:0Lrecommender_neural_network/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2C
Arecommender_neural_network/embedding_2/embeddings/Regularizer/Sum?
Crecommender_neural_network/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52E
Crecommender_neural_network/embedding_2/embeddings/Regularizer/mul/x?
Arecommender_neural_network/embedding_2/embeddings/Regularizer/mulMulLrecommender_neural_network/embedding_2/embeddings/Regularizer/mul/x:output:0Jrecommender_neural_network/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2C
Arecommender_neural_network/embedding_2/embeddings/Regularizer/mul?
IdentityIdentitySigmoid:y:0"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCallR^recommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOpT^recommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::::2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2?
Qrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOpQrecommender_neural_network/embedding/embeddings/Regularizer/Square/ReadVariableOp2?
Srecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOpSrecommender_neural_network/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
E__inference_embedding_3_layer_call_and_return_conditional_losses_9343

inputs	,
(embedding_lookup_readvariableop_resource
identity??embedding_lookup/ReadVariableOp?
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	?K*
dtype02!
embedding_lookup/ReadVariableOp?
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis?
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:?????????2
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:?????????2
embedding_lookup/Identity?
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs
?
n
(__inference_embedding_layer_call_fn_9484

inputs	
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_92622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????22

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0	?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?r
?	
user_embedding
	user_bias
movie_embedding

movie_bias
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
*J&call_and_return_all_conditional_losses
K_default_save_signature
L__call__"?
_tf_keras_model?{"class_name": "RecommenderNeuralNetwork", "name": "recommender_neural_network", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "RecommenderNeuralNetwork"}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
*M&call_and_return_all_conditional_losses
N__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 610, "output_dim": 50, "embeddings_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
*O&call_and_return_all_conditional_losses
P__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 610, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 9724, "output_dim": 50, "embeddings_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
?

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
*S&call_and_return_all_conditional_losses
T__call__"?
_tf_keras_layer?{"class_name": "Embedding", "name": "embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 9724, "output_dim": 1, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": null}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
?
iter

 beta_1

!beta_2
	"decay
#learning_ratemBmCmDmEvFvGvHvI"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
?
	variables
trainable_variables
$non_trainable_variables
regularization_losses

%layers
&metrics
'layer_regularization_losses
(layer_metrics
L__call__
K_default_save_signature
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
,
Wserving_default"
signature_map
B:@	?22/recommender_neural_network/embedding/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
U0"
trackable_list_wrapper
?
	variables
trainable_variables
)non_trainable_variables
regularization_losses

*layers
+metrics
,layer_regularization_losses
-layer_metrics
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
D:B	?21recommender_neural_network/embedding_1/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
.non_trainable_variables
regularization_losses

/layers
0metrics
1layer_regularization_losses
2layer_metrics
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
D:B	?K221recommender_neural_network/embedding_2/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
V0"
trackable_list_wrapper
?
	variables
trainable_variables
3non_trainable_variables
regularization_losses

4layers
5metrics
6layer_regularization_losses
7layer_metrics
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
D:B	?K21recommender_neural_network/embedding_3/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
	variables
trainable_variables
8non_trainable_variables
regularization_losses

9layers
:metrics
;layer_regularization_losses
<layer_metrics
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
=0"
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
'
U0"
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
'
V0"
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
?
	>total
	?count
@	variables
A	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
>0
?1"
trackable_list_wrapper
-
@	variables"
_generic_user_object
G:E	?226Adam/recommender_neural_network/embedding/embeddings/m
I:G	?28Adam/recommender_neural_network/embedding_1/embeddings/m
I:G	?K228Adam/recommender_neural_network/embedding_2/embeddings/m
I:G	?K28Adam/recommender_neural_network/embedding_3/embeddings/m
G:E	?226Adam/recommender_neural_network/embedding/embeddings/v
I:G	?28Adam/recommender_neural_network/embedding_1/embeddings/v
I:G	?K228Adam/recommender_neural_network/embedding_2/embeddings/v
I:G	?K28Adam/recommender_neural_network/embedding_3/embeddings/v
?2?
T__inference_recommender_neural_network_layer_call_and_return_conditional_losses_9407?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????	
?2?
__inference__wrapped_model_9239?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????	
?2?
9__inference_recommender_neural_network_layer_call_fn_9421?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????	
?2?
C__inference_embedding_layer_call_and_return_conditional_losses_9477?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_embedding_layer_call_fn_9484?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_1_layer_call_and_return_conditional_losses_9493?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_embedding_1_layer_call_fn_9500?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_2_layer_call_and_return_conditional_losses_9521?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_embedding_2_layer_call_fn_9528?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_embedding_3_layer_call_and_return_conditional_losses_9537?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_embedding_3_layer_call_fn_9544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_9555?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_9566?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
"__inference_signature_wrapper_9456input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_9239m0?-
&?#
!?
input_1?????????	
? "3?0
.
output_1"?
output_1??????????
E__inference_embedding_1_layer_call_and_return_conditional_losses_9493W+?(
!?
?
inputs?????????	
? "%?"
?
0?????????
? x
*__inference_embedding_1_layer_call_fn_9500J+?(
!?
?
inputs?????????	
? "???????????
E__inference_embedding_2_layer_call_and_return_conditional_losses_9521W+?(
!?
?
inputs?????????	
? "%?"
?
0?????????2
? x
*__inference_embedding_2_layer_call_fn_9528J+?(
!?
?
inputs?????????	
? "??????????2?
E__inference_embedding_3_layer_call_and_return_conditional_losses_9537W+?(
!?
?
inputs?????????	
? "%?"
?
0?????????
? x
*__inference_embedding_3_layer_call_fn_9544J+?(
!?
?
inputs?????????	
? "???????????
C__inference_embedding_layer_call_and_return_conditional_losses_9477W+?(
!?
?
inputs?????????	
? "%?"
?
0?????????2
? v
(__inference_embedding_layer_call_fn_9484J+?(
!?
?
inputs?????????	
? "??????????29
__inference_loss_fn_0_9555?

? 
? "? 9
__inference_loss_fn_1_9566?

? 
? "? ?
T__inference_recommender_neural_network_layer_call_and_return_conditional_losses_9407_0?-
&?#
!?
input_1?????????	
? "%?"
?
0?????????
? ?
9__inference_recommender_neural_network_layer_call_fn_9421R0?-
&?#
!?
input_1?????????	
? "???????????
"__inference_signature_wrapper_9456x;?8
? 
1?.
,
input_1!?
input_1?????????	"3?0
.
output_1"?
output_1?????????