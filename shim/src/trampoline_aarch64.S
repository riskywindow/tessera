/* Generic forwarding trampolines, AArch64. See trampoline_x86_64.S for what
 * these are and why they exist; this is the same machine in the other
 * instruction set.
 *
 * Register discipline on the slow path: x0-x7 carry integer arguments, x8 the
 * indirect result location, v0-v7 the floating-point ones, and x30 the
 * caller's return address. All of them are saved across the call to the
 * resolver and restored before the tail branch, so the real driver sees the
 * arguments and the return address the application passed. x15, x16 and x17
 * are the temporaries the procedure call standard already lets a callee
 * clobber.
 *
 * The counter is incremented with a load-exclusive/store-exclusive pair: the
 * build targets baseline armv8-a, where that is what the compiler's own
 * relaxed fetch_add on the same array compiles to.
 */

#if defined(__aarch64__)

	.text

.macro TESSERA_TRAMP name, idx
	.globl \name
	.type \name, %function
	.p2align 2
\name:
	.cfi_startproc
	hint #34			// bti c; a NOP where BTI is not enforced
	adrp x16, tessera_shim_counts+8*\idx
	add x16, x16, :lo12:(tessera_shim_counts+8*\idx)
1:
	ldxr x17, [x16]
	add x17, x17, #1
	stxr w15, x17, [x16]
	cbnz w15, 1b
	adrp x16, tessera_shim_slots+8*\idx
	ldr x17, [x16, :lo12:(tessera_shim_slots+8*\idx)]
	cbz x17, 2f
	br x17
2:
	/* 224 bytes: x29/x30, then v0-v7, then x0-x7 and x8. */
	stp x29, x30, [sp, #-224]!
	.cfi_def_cfa_offset 224
	.cfi_offset 29, -224
	.cfi_offset 30, -216
	mov x29, sp
	stp q0, q1, [sp, #16]
	stp q2, q3, [sp, #48]
	stp q4, q5, [sp, #80]
	stp q6, q7, [sp, #112]
	stp x0, x1, [sp, #144]
	stp x2, x3, [sp, #160]
	stp x4, x5, [sp, #176]
	stp x6, x7, [sp, #192]
	str x8, [sp, #208]
	mov w0, #\idx
	bl tessera_shim_resolve_tramp
	mov x16, x0
	ldp q0, q1, [sp, #16]
	ldp q2, q3, [sp, #48]
	ldp q4, q5, [sp, #80]
	ldp q6, q7, [sp, #112]
	ldp x0, x1, [sp, #144]
	ldp x2, x3, [sp, #160]
	ldp x4, x5, [sp, #176]
	ldp x6, x7, [sp, #192]
	ldr x8, [sp, #208]
	ldp x29, x30, [sp], #224
	.cfi_def_cfa_offset 0
	.cfi_restore 30
	.cfi_restore 29
	br x16
	.cfi_endproc
	.size \name, .-\name
.endm

#include "tessera_trampolines.inc"

/* Keep the library's branch-protection property when the rest of the objects
 * have it; without a matching note here the linker drops it for the whole
 * shared object. */
#if defined(__ARM_FEATURE_BTI_DEFAULT) || defined(__ARM_FEATURE_PAC_DEFAULT)
#if defined(__ARM_FEATURE_BTI_DEFAULT)
#define TESSERA_BTI_BIT 1
#else
#define TESSERA_BTI_BIT 0
#endif
#if defined(__ARM_FEATURE_PAC_DEFAULT)
#define TESSERA_PAC_BIT 2
#else
#define TESSERA_PAC_BIT 0
#endif
	.section .note.gnu.property,"a",%note
	.p2align 3
	.long 1f - 0f
	.long 4f - 1f
	.long 5			// NT_GNU_PROPERTY_TYPE_0
0:
	.asciz "GNU"
1:
	.p2align 3
	.long 0xc0000000	// GNU_PROPERTY_AARCH64_FEATURE_1_AND
	.long 3f - 2f		// pr_datasz
2:
	.long TESSERA_BTI_BIT | TESSERA_PAC_BIT
3:
	.p2align 3
4:
#endif

#endif /* __aarch64__ */

	.section .note.GNU-stack,"",%progbits
