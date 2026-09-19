/* Generic forwarding trampolines, x86-64.
 *
 * The shim exports every symbol the real driver exports (ADR-001): when the
 * shim IS libcuda.so.1, a symbol it does not define does not exist as far as
 * the application is concerned. Only a few dozen of those are hooked; the rest
 * are these trampolines, which need no knowledge of any signature because they
 * never touch the arguments -- they jump to the real function with the
 * caller's registers, stack and return address exactly as they arrived.
 *
 * One trampoline per unhooked symbol, generated into tessera_trampolines.inc
 * by shim/tools/gen_symbols.sh. Each one:
 *
 *   1. counts the call, with one atomic add on its own slot of
 *      tessera_shim_counts. That is what makes forwarded_unhooked a per-call
 *      number rather than a count of resolutions.
 *   2. loads its entry in tessera_shim_slots and, if it is set, TAIL-JUMPS
 *      there. No frame, no call: the real driver returns straight to the
 *      application.
 *   3. otherwise saves every register that can carry an argument -- the six
 *      integer registers, %rax (the vararg SSE count), and %xmm0-%xmm7 --
 *      calls the resolver with the symbol index, restores them, and tail-jumps
 *      to whatever it got back. The resolver always returns something
 *      callable, so this path cannot fall through.
 *
 * Both arrays are hidden symbols in this object, which is what lets these
 * PC-relative references link inside a shared library.
 */

#if defined(__x86_64__)

	.text

.macro TESSERA_TRAMP name, idx
	.globl \name
	.type \name, @function
	.p2align 4
\name:
	.cfi_startproc
	endbr64
	lock incq tessera_shim_counts+8*\idx(%rip)
	movq tessera_shim_slots+8*\idx(%rip), %r11
	testq %r11, %r11
	jz 9f
	jmp *%r11
9:
	/* 184 bytes: 8 * 16 for %xmm0-7, then 7 * 8 for the integer registers.
	   %rsp is 8 mod 16 on entry, so this both aligns it for the call and
	   makes the movaps offsets 16-byte aligned. */
	subq $184, %rsp
	.cfi_adjust_cfa_offset 184
	movaps %xmm0, (%rsp)
	movaps %xmm1, 16(%rsp)
	movaps %xmm2, 32(%rsp)
	movaps %xmm3, 48(%rsp)
	movaps %xmm4, 64(%rsp)
	movaps %xmm5, 80(%rsp)
	movaps %xmm6, 96(%rsp)
	movaps %xmm7, 112(%rsp)
	movq %rdi, 128(%rsp)
	movq %rsi, 136(%rsp)
	movq %rdx, 144(%rsp)
	movq %rcx, 152(%rsp)
	movq %r8, 160(%rsp)
	movq %r9, 168(%rsp)
	movq %rax, 176(%rsp)
	movl $\idx, %edi
	call tessera_shim_resolve_tramp
	movq %rax, %r11
	movaps (%rsp), %xmm0
	movaps 16(%rsp), %xmm1
	movaps 32(%rsp), %xmm2
	movaps 48(%rsp), %xmm3
	movaps 64(%rsp), %xmm4
	movaps 80(%rsp), %xmm5
	movaps 96(%rsp), %xmm6
	movaps 112(%rsp), %xmm7
	movq 128(%rsp), %rdi
	movq 136(%rsp), %rsi
	movq 144(%rsp), %rdx
	movq 152(%rsp), %rcx
	movq 160(%rsp), %r8
	movq 168(%rsp), %r9
	movq 176(%rsp), %rax
	addq $184, %rsp
	.cfi_adjust_cfa_offset -184
	jmp *%r11
	.cfi_endproc
	.size \name, .-\name
.endm

#include "tessera_trampolines.inc"

/* Keep the library's control-flow-enforcement property when the rest of the
 * objects have it; without a matching note here the linker would drop it for
 * the whole shared object. The endbr64 above is a NOP where it is not
 * enforced. */
#if defined(__CET__)
	.section .note.gnu.property,"a",@note
	.p2align 3
	.long 1f - 0f
	.long 4f - 1f
	.long 5			/* NT_GNU_PROPERTY_TYPE_0 */
0:
	.asciz "GNU"
1:
	.p2align 3
	.long 0xc0000002	/* GNU_PROPERTY_X86_FEATURE_1_AND */
	.long 3f - 2f		/* pr_datasz */
2:
	.long __CET__		/* bit 0 IBT, bit 1 SHSTK */
3:
	.p2align 3
4:
#endif

#endif /* __x86_64__ */

	.section .note.GNU-stack,"",@progbits
